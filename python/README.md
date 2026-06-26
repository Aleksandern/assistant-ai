# AssistantAI

Local assistant for live remote conversations.

Current project status:

- macOS audio capture from a selected app works through `ScreenCaptureKit`
- audio can be saved to `wav`
- `Silero VAD` is integrated for voice-activated utterance capture
- `whisper.cpp` transcription is available for local audio-file-to-text conversion
- OpenAI reply generation is available for text-to-text suggestions
- SQLite conversation storage is available for active conversations and conversation turns
- helper runner files exist for probing audio capture and saving one `wav` per utterance

## File Reuse For OpenAI Conversation Init

`python/modules/openai_conversation_initializer.py` initializes a new OpenAI-backed
live conversation and now reuses an existing OpenAI `file_id` when possible
instead of re-uploading the same local `.docx` on every run.

File source priority is:

1. local file `.docx`
2. `OPENAI_FILE_ID` from the repository-root `.env`
3. the latest stored `openai_file_id` from the local SQLite database

Behavior details:

- if a local file `.docx` is present, it is uploaded to OpenAI, the returned
  `file_id` is stored in the local conversation row, and the local `.docx` is
  deleted after the full flow succeeds
- if no local `.docx` is present, the initializer reuses
  `OPENAI_FILE_ID` from `.env` when available
- if neither a local `.docx` nor `.env` file id is available, the initializer
  reuses the latest non-empty `openai_file_id` stored in SQLite
- if all three sources are missing, conversation initialization fails
- `InitializedConversationRecord.file_name` is expected to be `None` on
  the `.env` and database reuse paths; runners treat that as normal reuse
  behavior, not as an error

Operational notes:

- the SQLite schema is expected to already include the
  `openai_file_id` column
- if the reused OpenAI `file_id` was deleted manually in OpenAI, future reuse
  attempts may stop working until a new local `.docx` is uploaded or a valid
  replacement `OPENAI_FILE_ID` is configured

## What To Install

The current project setup expects:

- macOS with `ScreenCaptureKit` support
- Python `3.13` or compatible Python `3`
- `ffmpeg`
- `.env` in the repository root for reply generation
- a Python virtual environment in `.venv`

## Installation

### 1. Install Homebrew dependencies

```bash
brew install ffmpeg
```

### 2. Set up the Python environment

From the repository root:

```bash
bash python/run/setup_python_env.sh
```

This script will:

- create `.venv` if needed
- upgrade `pip`
- install everything from `requirements.txt`

Installed Python packages currently include:

- `numpy`
- `openai`
- `pyobjc-core`
- `pyobjc-framework-ScreenCaptureKit`
- `silero-vad`

Additional notes:

- `silero-vad` also installs `torch` and `torchaudio`
- for package checks, module imports, and SDK inspection, use `python/.venv/bin/python`
- do not treat the system `python3` as the source of truth for the project's Python environment
- if `.venv` does not exist yet, establish that first before concluding that a package is missing

Example verification:

```bash
python/.venv/bin/python -c "import openai; print(openai.__version__)"
```

### 3. Install `whisper.cpp`

From the repository root:

```bash
bash python/run/setup_whisper_cpp.sh
```

This script will:

- clone `whisper.cpp` into `python/libs/whisper.cpp`
- build `whisper-cli`
- download the default `ggml-base.bin` model

## macOS Permissions

The terminal app that runs the runner files must have `Screen Recording` permission.

Steps:

1. Open `System Settings`
2. Go to `Privacy & Security`
3. Open `Screen Recording`
4. Enable access for your terminal app such as `Terminal` or `iTerm`
5. Restart that terminal app

Without this permission, app audio capture will fail.

## Project Layout

- `python/run/` contains runnable entrypoint files
- `python/modules/` contains the project's reusable code
- `python/libs/` is reserved for third-party source trees such as `whisper.cpp`

Important:

- code in `python/libs/` should not be edited directly
- normal Python dependencies are installed into `.venv`, not copied into `python/libs/`

## Current Runners

### List macOS capture targets

```bash
python/.venv/bin/python python/run/capture_system_audio_macos.py --list-targets
```

### Record audio from a selected app to WAV

Example:

```bash
python/.venv/bin/python python/run/capture_system_audio_macos.py --app-name Zoom --duration 10 --output python/artifacts/zoom.wav
```

### Voice-activated audio capture

This script monitors the target and uses `Silero VAD` to detect speech.

Current behavior:

- audio is captured from one continuous listener
- `Silero VAD` sees a sliding context window instead of one isolated chunk
- speech chunks are collected into one utterance
- short pauses inside speech are tolerated
- a small `pre-roll` protects the beginning of speech
- a small `post-roll` protects the tail
- the utterance file is finalized only after a longer pause

This means the script now tries to save one WAV per spoken utterance instead of saving rigid fixed-length chunks.

When these runners initialize an OpenAI conversation for the current session,
they follow the same file reuse flow described above. After the first
successful upload and local cleanup, subsequent runs normally reuse the stored
OpenAI `file_id` instead of requiring the local `.docx` to still be
present.

```bash
python/.venv/bin/python python/run/capture_voice_activated_macos.py --app-name Zoom
```

Useful options:

- `--segment-duration`
  Length of the internal capture chunk used by the listener.
- `--vad-threshold`
  Speech confidence threshold passed to `Silero VAD`.
- `--min-speech-ms`
  Minimum amount of detected speech required before an utterance is forwarded downstream.
- `--speech-pad-ms`
  Padding added by `Silero VAD` around detected speech.
- `--pre-roll-ms`
  Audio kept before the detected start of speech.
- `--max-pause-ms`
  Project-level setting that decides how much silence is tolerated before the current utterance WAV file is closed.
- `--post-roll-ms`
  Audio tail kept after the detected end of speech.

### ffmpeg audio probe

```bash
python/.venv/bin/python python/run/capture_audio_probe.py --help
```

### Transcribe one audio file to text

```bash
python/.venv/bin/python python/run/transcribe_audio_file.py python/artifacts/system-audio-test.wav
```

Useful options:

- `--language`
  Optional Whisper language code such as `en` or `ru`.
- `--model-path`
  Override the default `whisper.cpp` model path.
- `--whisper-cli-path`
  Override the default `whisper.cpp` CLI path.
- `--use-gpu`
  Opt in to GPU inference. By default the runner uses CPU mode for safer compatibility.

## OpenAI Reply Module

The reusable reply-generation module lives in `python/modules/openai_reply_generator.py`.

Example:

```python
from modules.openai_reply_generator import generate_chatgpt_reply

reply = generate_chatgpt_reply("What is your experience with Python?")
print(reply)
```

Behavior:

- takes one input text string
- sends that text to OpenAI Responses API
- returns plain text from the model reply

Configuration:

Use a repository-root `.env` file. Start from `.env.example`.

Expected variables:

```dotenv
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-5-mini
OPENAI_FILE_ID=file_abc123
OPENAI_REPLY_INSTRUCTIONS="..."
OPENAI_CONVERSATION_INSTRUCTIONS="..."
OPENAI_CONVERSATION_FILE_MESSAGE="..."
OPENAI_CONVERSATION_TOPIC_HINT_TEMPLATE="Conversation topic hint: {topic_hint}"
OPENAI_CONVERSATION_REPLY_INSTRUCTIONS="..."
OPENAI_TASK_SOLVER_PROMPT="..."
```

- `OPENAI_MODEL` is optional; default is `gpt-5-mini`
- `OPENAI_FILE_ID` is optional and is used as the first reuse source when
  no local file `.docx` is available
- OpenAI prompt and instruction texts are now configured through `.env`; see
  repository-root `.env.example` for the full documented set of keys
- Shared OpenAI env parsing now lives in `python/modules/openai_env_config.py`
  so the OpenAI modules use one consistent `.env` loading and validation path
- `api_key=...` can still be passed explicitly by orchestrators when needed

## SQLite Conversation Store Module

The reusable SQLite storage module lives in `python/modules/sqlite_conversation_store.py`.

It stores:

- conversations with a short `topic_hint`
- optional OpenAI conversation id and reused `file_id`
- conversation turns linked to a conversation
- remote speaker text
- optional translated remote text
- suggested reply text
- optional final reply text
- audio filename and parsed audio timestamp

The default database path is:

```text
database/assistantai.sqlite3
```

Example:

```python
from modules.sqlite_conversation_store import (
    add_turn_to_active_conversation,
    create_conversation,
)

conversation = create_conversation("Python conversation")

turn = add_turn_to_active_conversation(
    remote_text="Tell me about your Python experience.",
    remote_text_translate="Расскажите о вашем опыте с Python.",
    reply_text_suggest="I have used Python for backend services, automation, and data processing.",
    reply_text=None,
    audio_filename="utterance-conv-1-20260422-102530-000001.wav",
)

print(conversation)
print(turn)
```

## Current Limitation

The current utterance logic is already much better than fixed-length chunk saving, but it still depends on project-level thresholds such as `--pre-roll-ms`, `--post-roll-ms`, and `--max-pause-ms`. Different call apps and voices may need small tuning.
