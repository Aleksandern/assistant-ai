#!/usr/bin/env python3
"""
Continuous voice-activated macOS audio capture.

What this runner does:
- Opens one continuous ScreenCaptureKit audio stream for the selected app/display.
- Reads short analysis chunks from that live stream.
- Uses `Silero VAD` to decide whether each chunk contains speech.
- Keeps a short pre-roll buffer so the start of speech is not cut off.
- Keeps one utterance open across short pauses.
- Finalizes one WAV file only after a longer silence, keeping a small tail.

Important boundary:
- `Silero VAD` decides whether a chunk contains speech.
- This runner decides how multiple speech chunks become one utterance WAV file.
- Smaller analysis chunks reduce clipping at utterance boundaries.
- Use `--debug` to print listener and VAD diagnostics.
"""

from pathlib import Path
import sys

import numpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.captured_audio_queue import publish_captured_audio
from modules.macos_audio_listener import ContinuousAudioListener, list_targets, resolve_target
from modules.openai_conversation_initializer import initialize_openai_conversation
from modules.utterance_assembler import FinalizedUtterance, UtteranceAssembler
from modules.wav_file_writer import ensure_output_dir
from modules.wav_voice_activity import StreamingVoiceActivityDetector


def _format_file_source_status(file_name: str | None) -> str:
    if file_name:
        return f"file_name={file_name}"
    return "file_name=None (reused existing OpenAI file)"


def main(argv: list[str] | None = None) -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Continuously monitor a macOS app/display and save utterance WAV files using Silero VAD."
    )
    parser.add_argument("--display-index", type=int, default=0, help="Display index from shareable content.")
    parser.add_argument("--app-name", help="Monitor an application whose visible name matches the given text.")
    parser.add_argument("--bundle-id", help="Monitor an application by exact macOS bundle identifier.")
    parser.add_argument("--list-targets", action="store_true", help="List shareable displays and applications.")
    parser.add_argument("--segment-duration", type=float, default=0.2, help="Length of each analysis chunk in seconds. Smaller values reduce clipping at utterance boundaries.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target mono sample rate for VAD and saved WAV files.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts/voice-activated"), help="Directory for finalized utterance WAV files.")
    parser.add_argument("--prefix", default="utterance", help="Prefix for saved utterance files.")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Speech confidence threshold used by Silero VAD.")
    parser.add_argument("--min-speech-ms", type=int, default=250, help="Minimum speech duration passed to Silero VAD.")
    parser.add_argument("--speech-pad-ms", type=int, default=30, help="Padding added by Silero VAD around detected speech regions.")
    parser.add_argument("--pre-roll-ms", type=int, default=400, help="Audio kept before detected speech so the start is not cut off.")
    parser.add_argument("--max-pause-ms", type=int, default=800, help="Maximum tolerated pause inside one utterance before the file is finalized.")
    parser.add_argument("--post-roll-ms", type=int, default=300, help="Silence tail kept at the end of a finalized utterance.")
    parser.add_argument("--poll-interval", type=float, default=0.05, help="Short pause while waiting during fully silent periods.")
    parser.add_argument("--include-self-audio", action="store_true", help="Include audio from the current process.")
    parser.add_argument("--timeout", type=float, default=8.0, help="Timeout for ScreenCaptureKit async operations.")
    parser.add_argument("--debug", action="store_true", help="Print listener and VAD debug information.")
    parser.add_argument(
        "--log-detected-speech-chunks",
        action="store_true",
        help="Print one line for each chunk classified as speech inside an utterance.",
    )
    parser.add_argument(
        "--log-silence-inside-utterance",
        action="store_true",
        help="Print one line for each silent chunk accumulated inside an open utterance.",
    )
    parser.add_argument(
        "--log-discarded-silent-chunks",
        action="store_true",
        help="Print one line for each fully silent chunk discarded before speech starts.",
    )
    args = parser.parse_args(argv)

    try:
        if args.list_targets:
            list_targets(timeout=args.timeout)
            return 0

        output_dir = ensure_output_dir(args.output_dir)
        listener = ContinuousAudioListener(
            display_index=args.display_index,
            app_name=args.app_name,
            bundle_id=args.bundle_id,
            include_self_audio=args.include_self_audio,
            timeout=args.timeout,
            sample_rate=args.sample_rate,
            debug=args.debug,
        )
        resolve_target(
            display_index=args.display_index,
            app_name=args.app_name,
            bundle_id=args.bundle_id,
            timeout=args.timeout,
        )
        initialized_conversation = initialize_openai_conversation(
            "",
        )
        print(
            "Conversation initialized:",
            f"conversation_id={initialized_conversation.conversation_id}",
            f"openai_conversation_id={initialized_conversation.openai_conversation_id}",
            _format_file_source_status(initialized_conversation.file_name),
        )
        target_description = listener.start()
        print(f"Voice-activated capture started. Listening to {target_description}. Press Ctrl+C to stop.")

        assembler = UtteranceAssembler(
            sample_rate=args.sample_rate,
            segment_duration=args.segment_duration,
            pre_roll_ms=args.pre_roll_ms,
            max_pause_ms=0,
            post_roll_ms=args.post_roll_ms,
        )
        streaming_vad = StreamingVoiceActivityDetector(
            sample_rate=args.sample_rate,
            threshold=args.vad_threshold,
            min_silence_duration_ms=args.max_pause_ms,
            speech_pad_ms=args.speech_pad_ms,
        )

        def publish_finalized_utterance(finalized_utterance: FinalizedUtterance) -> None:
            speech_ms = finalized_utterance.speech_chunk_count * assembler.chunk_ms
            if speech_ms < args.min_speech_ms:
                if args.debug:
                    print(
                        "[runner] discarded short utterance",
                        f"speech_ms={speech_ms}",
                        f"min_speech_ms={args.min_speech_ms}",
                    )
                return
            final_path = publish_captured_audio(
                audio=finalized_utterance.audio,
                sample_rate=args.sample_rate,
                storage_dir=output_dir,
                conversation_id=initialized_conversation.conversation_id,
                prefix=args.prefix,
            )
            print(
                f"Saved utterance: {final_path.name} "
                f"(chunks={finalized_utterance.utterance_chunk_count + finalized_utterance.trailing_chunk_count}, "
                f"trailing_pause_ms={finalized_utterance.trailing_pause_ms})"
            )

        def finalize_utterance() -> None:
            finalized_utterance = assembler.finalize()
            if finalized_utterance is None:
                return
            publish_finalized_utterance(finalized_utterance)

        try:
            while True:
                audio_chunk = listener.read_chunk(args.segment_duration)
                if audio_chunk is None:
                    if args.debug:
                        print("[runner] listener returned no chunk", listener.debug_state())
                    break

                vad_result = streaming_vad.process_chunk(audio_chunk)
                if args.debug:
                    peak = float(numpy.max(numpy.abs(audio_chunk))) if audio_chunk.size else 0.0
                    rms = float(numpy.sqrt(numpy.mean(audio_chunk * audio_chunk))) if audio_chunk.size else 0.0
                    print(
                        "[runner] analyzed chunk",
                        f"samples={audio_chunk.shape[0]}",
                        f"speech_active={vad_result.speech_active}",
                        f"speech_started={vad_result.speech_started}",
                        f"speech_ended={vad_result.speech_ended}",
                        f"contains_speech={vad_result.contains_speech}",
                        f"processed_samples={vad_result.processed_samples}",
                        f"peak={peak:.4f}",
                        f"rms={rms:.4f}",
                        f"vad_pending_samples={streaming_vad.pending_sample_count()}",
                    )

                result = assembler.push_chunk(audio_chunk, has_recent_voice=vad_result.contains_speech)
                if result.event == "speech":
                    if args.debug and result.prepended_pre_roll_samples:
                        print("[runner] prepended pre-roll samples", result.prepended_pre_roll_samples)
                    if args.log_detected_speech_chunks:
                        print("Detected speech chunk")
                elif result.event == "silence":
                    if args.log_silence_inside_utterance:
                        print(f"Silence inside utterance (accumulated_pause_ms={result.accumulated_pause_ms})")
                    if result.finalized_utterance is not None:
                        publish_finalized_utterance(result.finalized_utterance)
                else:
                    if args.log_discarded_silent_chunks:
                        print("Discarded silent chunk")
                    time.sleep(args.poll_interval)
        finally:
            if args.debug:
                print("[runner] final listener state", listener.debug_state())
            listener.stop()
    except KeyboardInterrupt:
        if "finalize_utterance" in locals():
            finalize_utterance()
        print("\nStopped by user.")
        return 0
    except Exception as exc:
        print(f"Voice-activated capture failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
