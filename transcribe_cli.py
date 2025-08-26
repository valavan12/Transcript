#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import asyncio
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Speech-to-text (local, fast, accurate)
# Requires ffmpeg installed on system for many formats
try:
	from faster_whisper import WhisperModel  # type: ignore
except Exception as exc:  # pragma: no cover
	print("Error: faster-whisper is required. Install with: pip install faster-whisper", file=sys.stderr)
	raise

# PydanticAI for summarization
try:
	from pydantic_ai import Agent
	from pydantic_ai.models.openai import OpenAIChatModel  # type: ignore
	from pydantic_ai.models.mistral import MistralModel  # type: ignore
	from pydantic_ai.models.google import GoogleModel  # type: ignore
except Exception as exc:  # pragma: no cover
	print("Error: pydantic-ai and provider extras are required. Install with: pip install pydantic-ai openai mistralai google-generativeai", file=sys.stderr)
	raise


SUPPORTED_PROVIDERS = {"openai", "mistral", "gemini"}
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_MISTRAL_MODEL = "mistral-large-latest"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


@dataclass
class Segment:
	start: float
	end: float
	text: str
	words: Optional[List[Dict[str, Any]]] = None


@dataclass
class TranscriptionResult:
	text: str
	language: Optional[str]
	segments: List[Segment]
	duration: Optional[float]


def transcribe_audio(
	file_path: Path,
	model_size: str = "medium",
	device: Optional[str] = None,
	compute_type: str = "int8_float16",
	vad_filter: bool = True,
	beam_size: int = 5,
	best_of: int = 5,
	word_timestamps: bool = False,
	temperature: float = 0.0,
	language: Optional[str] = None,
) -> TranscriptionResult:
	"""Transcribe audio with Faster-Whisper and return full text plus segments."""
	if not file_path.exists():
		raise FileNotFoundError(f"Audio file not found: {file_path}")

	# Initialize model
	# device: "cpu", "cuda", or None to auto-pick
	model = WhisperModel(
		model_size,
		device=device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"),
		compute_type=compute_type,
	)

	start_time = time.time()
	segments_iter, info = model.transcribe(
		str(file_path),
		vad_filter=vad_filter,
		beam_size=beam_size,
		best_of=best_of,
		word_timestamps=word_timestamps,
		temperature=temperature,
		language=language,
	)

	segments: List[Segment] = []
	collected_text_parts: List[str] = []
	for seg in segments_iter:
		seg_text = seg.text or ""
		collected_text_parts.append(seg_text.strip())
		word_items = None
		if word_timestamps and getattr(seg, "words", None):
			word_items = [
				{"start": w.start, "end": w.end, "word": w.word}
				for w in seg.words
			]
		segments.append(
			Segment(start=float(seg.start or 0.0), end=float(seg.end or 0.0), text=seg_text.strip(), words=word_items)
		)

	full_text = "\n".join([t for t in collected_text_parts if t])
	elapsed = time.time() - start_time

	return TranscriptionResult(
		text=full_text,
		language=getattr(info, "language", None),
		segments=segments,
		duration=elapsed,
	)


def build_agent(provider: str, model_name: Optional[str]) -> Agent:
	provider_lower = provider.lower()
	if provider_lower not in SUPPORTED_PROVIDERS:
		raise ValueError(f"Unsupported provider '{provider}'. Choose from: {', '.join(sorted(SUPPORTED_PROVIDERS))}")

	if provider_lower == "openai":
		model = OpenAIChatModel(model_name or DEFAULT_OPENAI_MODEL)
	elif provider_lower == "mistral":
		model = MistralModel(model_name or DEFAULT_MISTRAL_MODEL)
	elif provider_lower == "gemini":
		model = GoogleModel(model_name or DEFAULT_GEMINI_MODEL)
	else:  # pragma: no cover - guarded above
		raise ValueError("Unexpected provider")

	return Agent(
		model=model,
		system_prompt=(
			"You are an expert meeting summarizer. Produce a faithful, concise summary with:\n"
			"- Title\n- Participants (if present)\n- Key points as bullet list\n- Decisions\n- Action items (owner, due if stated)\n- Risks and open questions\n\n"
			"Write in clear, neutral tone. Keep to the facts."
		),
	)


def summarize_text(agent: Agent, transcript: str, max_chars: int = 16000) -> str:
	"""Summarize transcript text using the provided Agent."""
	if not transcript.strip():
		return "(No transcript text)"

	# Truncate to a safe context length
	text = transcript[:max_chars]

	# Support both sync and async Agent.run across pydantic-ai versions
	if hasattr(agent, "run_sync"):
		resp = agent.run_sync(f"Summarize this conversation transcript.\n\nTranscript:\n{text}")
	else:
		maybe_coro = agent.run(f"Summarize this conversation transcript.\n\nTranscript:\n{text}")
		resp = asyncio.run(maybe_coro) if inspect.isawaitable(maybe_coro) else maybe_coro

	# Try common response attributes across pydantic-ai versions
	for attr in ("text", "output_text", "content", "data"):
		val = getattr(resp, attr, None)
		if isinstance(val, str) and val.strip():
			return val.strip()

	# Fallbacks: val might be non-string but printable (e.g., dict)
	for attr in ("content", "data"):
		val = getattr(resp, attr, None)
		if val is not None:
			return str(val).strip()

	return str(resp).strip() or "(No summary returned)"


def write_outputs(
	output_dir: Path,
	base_name: str,
	transcription: TranscriptionResult,
	summary_md: str,
	provider: str,
	model_used: Optional[str],
	in_path: Path,
) -> Dict[str, Any]:
	output_dir.mkdir(parents=True, exist_ok=True)

	transcript_path = output_dir / f"{base_name}.transcript.txt"
	summary_path = output_dir / f"{base_name}.summary.md"
	metadata_path = output_dir / f"{base_name}.metadata.json"
	segments_path = output_dir / f"{base_name}.segments.json"

	transcript_path.write_text(transcription.text or "", encoding="utf-8")
	summary_path.write_text(summary_md or "", encoding="utf-8")

	segments_payload = [asdict(s) for s in transcription.segments]
	segments_path.write_text(json.dumps(segments_payload, ensure_ascii=False, indent=2), encoding="utf-8")

	metadata = {
		"input_file": str(in_path.resolve()),
		"provider": provider,
		"model": model_used,
		"language": transcription.language,
		"duration_seconds": transcription.duration,
		"transcript_chars": len(transcription.text or ""),
		"summary_chars": len(summary_md or ""),
		"created_at": int(time.time()),
	}
	metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
	return {
		"transcript": str(transcript_path),
		"summary": str(summary_path),
		"metadata": str(metadata_path),
		"segments": str(segments_path),
	}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Transcribe an audio file and summarize it using PydanticAI with selectable provider.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument("audio", type=str, help="Path to audio file (.wav, .mp3, .m4a, etc.)")
	parser.add_argument("--provider", choices=sorted(SUPPORTED_PROVIDERS), default="openai", help="LLM provider for summary")
	parser.add_argument("--model", type=str, default=None, help="LLM model name override")
	parser.add_argument("--whisper-model", type=str, default="medium", help="Faster-Whisper model size (e.g., tiny, base, small, medium, large-v3)")
	parser.add_argument("--device", type=str, default=None, help="Device for Faster-Whisper (cpu/cuda). Auto if omitted")
	parser.add_argument("--compute-type", type=str, default="int8_float16", help="Compute type for Faster-Whisper")
	parser.add_argument("--no-vad", action="store_true", help="Disable VAD filtering")
	parser.add_argument("--word-timestamps", action="store_true", help="Include per-word timestamps in segments")
	parser.add_argument("--language", type=str, default=None, help="Force language code for transcription (e.g., en, fr)")
	parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to write outputs")
	parser.add_argument("--max-chars", type=int, default=16000, help="Max characters of transcript to summarize")
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)
	in_path = Path(args.audio)
	if not in_path.exists():
		print(f"Error: input file not found: {in_path}", file=sys.stderr)
		return 2

	# Transcribe
	try:
		transcription = transcribe_audio(
			in_path,
			model_size=args.whisper_model,
			device=args.device,
			compute_type=args.compute_type,
			vad_filter=not args.no_vad,
			word_timestamps=args.word_timestamps,
			language=args.language,
		)
	except Exception as exc:
		print(f"Transcription failed: {exc}", file=sys.stderr)
		return 3

	# Summarize
	try:
		agent = build_agent(args.provider, args.model)
		summary_md = summarize_text(agent, transcription.text, max_chars=args.max_chars)
	except Exception as exc:
		print(f"Summarization failed: {exc}", file=sys.stderr)
		return 4

	# Write outputs
	try:
		output_dir = Path(args.output_dir)
		base_name = in_path.stem
		paths = write_outputs(
			output_dir=output_dir,
			base_name=base_name,
			transcription=transcription,
			summary_md=summary_md,
			provider=args.provider,
			model_used=args.model,
			in_path=in_path,
		)
	except Exception as exc:
		print(f"Writing outputs failed: {exc}", file=sys.stderr)
		return 5

	print("Success. Files written:")
	for key, val in paths.items():
		print(f"- {key}: {val}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
