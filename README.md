# AssistantAI

A real-time desktop AI assistant for macOS.

AssistantAI listens to a selected application (Zoom, Teams, Telegram, browsers, and other apps), detects spoken conversations, generates context-aware responses using an attached document and can analyze screenshots through its built-in web interface.

Built as a personal engineering project to explore low-latency desktop AI, speech processing, context-aware conversations and screenshot analysis.

## Current capabilities include:

- Application-specific audio capture using ScreenCaptureKit
- Voice activity detection (Silero VAD)
- Local speech recognition with whisper.cpp
- Context-aware conversations using an attached document
- Screenshot capture and AI analysis
- Local conversation history
- Remote web interface

The current implementation is written in Python.

Useful starting points:

- Python setup: `bash python/run/setup_python_env.sh`
- Python docs: `python/README.md`
- Audio capture research: `docs/modules/audio-capture-research.md`

Additional implementation details, installation instructions, and OpenAI configuration are available in python/README.md.

Future versions may include additional components and languages as the project evolves.

## Roadmap

- Additional LLM backends (Ollama, llama.cpp)
- Additional document formats
- Improved screenshot analysis
