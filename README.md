# Transcript CLI

Process an audio file: transcribe with Faster-Whisper, then summarize via PydanticAI using your chosen provider (OpenAI, Mistral, or Gemini Flash).

## Prerequisites
- Python 3.10+
- FFmpeg installed and in PATH
- API keys as needed for your chosen provider
  - `OPENAI_API_KEY` for OpenAI
  - `MISTRAL_API_KEY` for Mistral
  - `GOOGLE_API_KEY` for Gemini

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python transcribe_cli.py /path/to/audio.m4a --provider openai
```
Options:
- `--provider` one of `openai`, `mistral`, `gemini` (default: `openai`)
- `--model` override model name (e.g., `gpt-4o-mini`, `mistral-large-latest`, `gemini-1.5-flash`)
- `--whisper-model` Faster-Whisper size (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `--device` `cpu` or `cuda`
- `--compute-type` e.g. `int8_float16` (default)
- `--no-vad` disable VAD filter
- `--word-timestamps` include per-word timestamps
- `--language` force language (e.g., `en`)
- `--output-dir` where to write files (default `./outputs`)
- `--max-chars` transcript chars to summarize

## Output
Writes four files into `output-dir` with the base name of the audio file:
- `<name>.transcript.txt`
- `<name>.summary.md`
- `<name>.segments.json` (timestamped segments)
- `<name>.metadata.json`

## Environment
Set API keys for provider you use:
```bash
export OPENAI_API_KEY=...       # OpenAI
export MISTRAL_API_KEY=...      # Mistral
export GOOGLE_API_KEY=...       # Gemini
```

## Notes
- Faster-Whisper requires FFmpeg for many formats.
- Use `--whisper-model large-v3` for best accuracy if you have GPU VRAM.
- Summarization truncates very long transcripts via `--max-chars`.
