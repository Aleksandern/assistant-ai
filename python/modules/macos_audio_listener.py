from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import ctypes
import traceback
from pathlib import Path

import numpy
import objc
import ScreenCaptureKit
import CoreMedia
import torch
import torchaudio.functional as audio_functional
from Foundation import NSDate, NSRunLoop, NSObject, NSURL


def ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found in PATH. Install ffmpeg first.")
    return ffmpeg


def pump_runloop(seconds: float) -> None:
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(seconds))


def wait_for_event(event: threading.Event, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if event.wait(0.05):
            return True
        pump_runloop(0.05)
    return event.is_set()


def get_shareable_content(timeout: float):
    state: dict[str, object] = {}
    event = threading.Event()

    def handler(content, error):
        state["content"] = content
        state["error"] = error
        event.set()

    ScreenCaptureKit.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
        False,
        True,
        handler,
    )

    if not wait_for_event(event, timeout):
        raise RuntimeError(
            "Timed out waiting for ScreenCaptureKit shareable content. "
            "Check Screen Recording permission in macOS settings."
        )

    if state.get("error") is not None:
        raise RuntimeError(f"ScreenCaptureKit returned an error: {state['error']}")

    return state.get("content")


def list_targets(timeout: float) -> None:
    content = get_shareable_content(timeout=timeout)
    displays = list(content.displays())
    applications = list(content.applications())

    print("Displays:")
    for idx, display in enumerate(displays):
        print(f"  [{idx}] id={display.displayID()} size={display.width()}x{display.height()}")

    print("Applications:")
    for idx, app in enumerate(applications):
        name = app.applicationName() or "<unknown>"
        bundle_id = app.bundleIdentifier() or "<no-bundle-id>"
        print(f"  [{idx}] {name}  bundle={bundle_id}  pid={app.processID()}")


def resolve_target(*, display_index: int, app_name: str | None, bundle_id: str | None, timeout: float):
    content = get_shareable_content(timeout=timeout)
    displays = list(content.displays())
    applications = list(content.applications())

    if not displays:
        raise RuntimeError("ScreenCaptureKit returned no displays.")
    if display_index < 0 or display_index >= len(displays):
        raise RuntimeError(f"Display index {display_index} is out of range. Found {len(displays)} display(s).")

    display = displays[display_index]
    app = None

    if bundle_id:
        for candidate in applications:
            if (candidate.bundleIdentifier() or "").lower() == bundle_id.lower():
                app = candidate
                break
        if app is None:
            raise RuntimeError(f"No shareable application matched bundle id '{bundle_id}'.")
    elif app_name:
        normalized = app_name.lower()
        for candidate in applications:
            if normalized in (candidate.applicationName() or "").lower():
                app = candidate
                break
        if app is None:
            raise RuntimeError(f"No shareable application matched name '{app_name}'.")

    return display, app


def describe_target(app) -> str:
    if app is None:
        return "display audio"
    return (
        f"application audio: {app.applicationName() or '<unknown>'} "
        f"bundle={app.bundleIdentifier() or '<no-bundle-id>'} pid={app.processID()}"
    )


class RecordingDelegate(NSObject):
    def init(self):
        self = objc.super(RecordingDelegate, self).init()
        if self is None:
            return None
        self.started = threading.Event()
        self.finished = threading.Event()
        self.failed_error = None
        return self

    def recordingOutputDidStartRecording_(self, recording_output):
        self.started.set()

    def recordingOutputDidFinishRecording_(self, recording_output):
        self.finished.set()

    def recordingOutput_didFailWithError_(self, recording_output, error):
        self.failed_error = error
        self.finished.set()


def build_stream(display, app, movie_path: Path, exclude_current_process_audio: bool):
    if app is not None:
        content_filter = ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_includingApplications_exceptingWindows_(
            display,
            [app],
            [],
        )
    else:
        content_filter = ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            display,
            [],
        )

    config = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
    config.setCapturesAudio_(True)
    config.setExcludesCurrentProcessAudio_(exclude_current_process_audio)
    config.setCaptureMicrophone_(False)
    config.setWidth_(2)
    config.setHeight_(2)
    config.setQueueDepth_(3)

    delegate = RecordingDelegate.alloc().init()
    recording_config = ScreenCaptureKit.SCRecordingOutputConfiguration.alloc().init()
    recording_config.setOutputURL_(NSURL.fileURLWithPath_(str(movie_path)))
    recording_config.setOutputFileType_("com.apple.quicktime-movie")
    recording_output = ScreenCaptureKit.SCRecordingOutput.alloc().initWithConfiguration_delegate_(
        recording_config,
        delegate,
    )

    stream = ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(
        content_filter,
        config,
        None,
    )

    ok, error = stream.addRecordingOutput_error_(recording_output, None)
    if not ok:
        raise RuntimeError(f"Could not attach recording output: {error}")

    return stream, delegate


def start_stream(stream, timeout: float) -> None:
    state: dict[str, object] = {}
    event = threading.Event()

    def completion(error):
        state["error"] = error
        event.set()

    stream.startCaptureWithCompletionHandler_(completion)
    if not wait_for_event(event, timeout):
        raise RuntimeError("Timed out waiting for ScreenCaptureKit stream start.")
    if state.get("error") is not None:
        raise RuntimeError(f"Could not start capture: {state['error']}")


def stop_stream(stream, timeout: float) -> None:
    state: dict[str, object] = {}
    event = threading.Event()

    def completion(error):
        state["error"] = error
        event.set()

    stream.stopCaptureWithCompletionHandler_(completion)
    if not wait_for_event(event, timeout):
        raise RuntimeError("Timed out waiting for ScreenCaptureKit stream stop.")
    if state.get("error") is not None:
        raise RuntimeError(f"Could not stop capture: {state['error']}")


def capture_segment_to_movie(
    *,
    duration: int,
    display_index: int,
    app_name: str | None,
    bundle_id: str | None,
    include_self_audio: bool,
    timeout: float,
    announce_target: bool = True,
) -> tuple[Path, str]:
    display, app = resolve_target(
        display_index=display_index,
        app_name=app_name,
        bundle_id=bundle_id,
        timeout=timeout,
    )
    target_description = describe_target(app)
    if announce_target:
        print(f"Capturing {target_description}")

    movie_fd, movie_name = tempfile.mkstemp(prefix="assistantai-", suffix=".mov")
    os.close(movie_fd)
    Path(movie_name).unlink(missing_ok=True)
    movie_path = Path(movie_name)

    stream, delegate = build_stream(
        display=display,
        app=app,
        movie_path=movie_path,
        exclude_current_process_audio=not include_self_audio,
    )

    try:
        start_stream(stream, timeout=timeout)
        if not wait_for_event(delegate.started, timeout=timeout):
            raise RuntimeError("Recording output never reported a start event.")

        print(f"Recording system audio for {duration} second(s)...")
        end_at = time.time() + duration
        while time.time() < end_at:
            pump_runloop(0.1)

        stop_stream(stream, timeout=timeout)
        wait_for_event(delegate.finished, timeout=2.0)

        if delegate.failed_error is not None:
            raise RuntimeError(f"Recording failed: {delegate.failed_error}")
        if not movie_path.exists():
            raise RuntimeError("Expected ScreenCaptureKit recording file was not created.")
        return movie_path, target_description
    except Exception:
        movie_path.unlink(missing_ok=True)
        raise


class _AudioChunkCollector(NSObject):
    def initWithTargetSampleRate_debug_(self, target_sample_rate, debug):
        self = objc.super(_AudioChunkCollector, self).init()
        if self is None:
            return None
        self.target_sample_rate = int(target_sample_rate)
        self.debug = bool(debug)
        self.condition = threading.Condition()
        self.chunks: list[numpy.ndarray] = []
        self.closed = False
        self.callback_count = 0
        self.last_error = None
        self.last_chunk_samples = 0
        self.last_input_sample_rate = None
        self.last_channels = None
        return self

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def pop_chunk(self, duration_seconds: float) -> numpy.ndarray | None:
        needed_samples = max(1, int(round(duration_seconds * self.target_sample_rate)))
        with self.condition:
            while True:
                available = sum(chunk.shape[0] for chunk in self.chunks)
                if available >= needed_samples:
                    break
                if self.closed:
                    return None
                self.condition.wait(timeout=0.2)

            output_parts = []
            remaining = needed_samples
            while remaining > 0 and self.chunks:
                head = self.chunks[0]
                if head.shape[0] <= remaining:
                    output_parts.append(head)
                    self.chunks.pop(0)
                    remaining -= head.shape[0]
                else:
                    output_parts.append(head[:remaining])
                    self.chunks[0] = head[remaining:]
                    remaining = 0

        if not output_parts:
            return None
        return numpy.concatenate(output_parts)

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
        if output_type != ScreenCaptureKit.SCStreamOutputTypeAudio:
            return
        try:
            audio, meta = _extract_audio_chunk(sample_buffer, self.target_sample_rate)
            self.callback_count += 1
            self.last_chunk_samples = 0 if audio is None else int(audio.shape[0])
            self.last_input_sample_rate = meta["input_sample_rate"]
            self.last_channels = meta["channels"]
            if self.debug and self.callback_count <= 5:
                print(
                    "[listener] audio callback",
                    f"count={self.callback_count}",
                    f"input_rate={meta['input_sample_rate']}",
                    f"target_rate={self.target_sample_rate}",
                    f"channels={meta['channels']}",
                    f"bits={meta['bits_per_channel']}",
                    f"float={meta['is_float']}",
                    f"samples={self.last_chunk_samples}",
                )
            if audio is None or audio.shape[0] == 0:
                return
            with self.condition:
                self.chunks.append(audio)
                self.condition.notify_all()
        except Exception as exc:
            self.last_error = str(exc)
            if self.debug:
                print(f"[listener] audio callback error: {exc}")
                traceback.print_exc()
            return


def _asbd_field(asbd, index: int, name: str):
    if hasattr(asbd, name):
        return getattr(asbd, name)
    return asbd[index]


def _extract_audio_chunk(sample_buffer, target_sample_rate: int) -> tuple[numpy.ndarray | None, dict[str, int | bool]]:
    format_description = CoreMedia.CMSampleBufferGetFormatDescription(sample_buffer)
    if format_description is None:
        return None, {
            "input_sample_rate": 0,
            "channels": 0,
            "bits_per_channel": 0,
            "is_float": False,
        }

    asbd = CoreMedia.CMAudioFormatDescriptionGetStreamBasicDescription(format_description)
    sample_rate = int(round(_asbd_field(asbd, 0, "mSampleRate")))
    format_flags = int(_asbd_field(asbd, 2, "mFormatFlags"))
    bytes_per_frame = int(_asbd_field(asbd, 5, "mBytesPerFrame"))
    channels = int(_asbd_field(asbd, 6, "mChannelsPerFrame"))
    bits_per_channel = int(_asbd_field(asbd, 7, "mBitsPerChannel"))

    data_buffer = CoreMedia.CMSampleBufferGetDataBuffer(sample_buffer)
    if data_buffer is None:
        return None, {
            "input_sample_rate": sample_rate,
            "channels": channels,
            "bits_per_channel": bits_per_channel,
            "is_float": is_float,
        }

    byte_count = int(CoreMedia.CMBlockBufferGetDataLength(data_buffer))
    if byte_count <= 0:
        return None, {
            "input_sample_rate": sample_rate,
            "channels": channels,
            "bits_per_channel": bits_per_channel,
            "is_float": is_float,
        }

    raw = (ctypes.c_char * byte_count)()
    status = CoreMedia.CMBlockBufferCopyDataBytes(data_buffer, 0, byte_count, raw)
    copied_buffer = raw
    if isinstance(status, tuple):
        error_code = status[0]
        if len(status) > 1 and status[1] is not None:
            copied_buffer = status[1]
    else:
        error_code = status

    if error_code != 0:
        raise RuntimeError(f"CMBlockBufferCopyDataBytes failed with status {error_code}")

    pcm_bytes = bytes(copied_buffer)
    sample_width = max(1, bits_per_channel // 8)
    is_float = bool(format_flags & 0x1)
    is_interleaved = bytes_per_frame > sample_width

    if is_float and bits_per_channel == 32:
        samples = numpy.frombuffer(pcm_bytes, dtype=numpy.float32)
    elif not is_float and bits_per_channel == 16:
        samples = numpy.frombuffer(pcm_bytes, dtype=numpy.int16).astype(numpy.float32) / 32768.0
    else:
        raise RuntimeError(
            f"Unsupported audio format from ScreenCaptureKit: bits={bits_per_channel}, float={is_float}"
        )

    if channels > 1:
        if is_interleaved:
            samples = samples.reshape(-1, channels).mean(axis=1)
        else:
            frames = samples.shape[0] // channels
            samples = samples[: frames * channels].reshape(channels, frames).mean(axis=0)

    audio = torch.from_numpy(samples.copy())
    if sample_rate != target_sample_rate:
        audio = audio_functional.resample(audio, sample_rate, target_sample_rate)
    return audio.numpy().astype(numpy.float32, copy=False), {
        "input_sample_rate": sample_rate,
        "channels": channels,
        "bits_per_channel": bits_per_channel,
        "is_float": is_float,
    }


class ContinuousAudioListener:
    def __init__(
        self,
        *,
        display_index: int,
        app_name: str | None,
        bundle_id: str | None,
        include_self_audio: bool,
        timeout: float,
        sample_rate: int,
        debug: bool = False,
    ):
        self.display_index = display_index
        self.app_name = app_name
        self.bundle_id = bundle_id
        self.include_self_audio = include_self_audio
        self.timeout = timeout
        self.sample_rate = sample_rate
        self.debug = debug
        self.stream = None
        self.collector = None

    def start(self) -> str:
        display, app = resolve_target(
            display_index=self.display_index,
            app_name=self.app_name,
            bundle_id=self.bundle_id,
            timeout=self.timeout,
        )
        target_description = describe_target(app)

        if app is not None:
            content_filter = ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_includingApplications_exceptingWindows_(
                display,
                [app],
                [],
            )
        else:
            content_filter = ScreenCaptureKit.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                display,
                [],
            )

        config = ScreenCaptureKit.SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        config.setExcludesCurrentProcessAudio_(not self.include_self_audio)
        config.setCaptureMicrophone_(False)
        config.setWidth_(2)
        config.setHeight_(2)
        config.setQueueDepth_(8)

        self.collector = _AudioChunkCollector.alloc().initWithTargetSampleRate_debug_(self.sample_rate, self.debug)
        self.stream = ScreenCaptureKit.SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter,
            config,
            None,
        )
        ok, error = self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self.collector,
            ScreenCaptureKit.SCStreamOutputTypeAudio,
            None,
            None,
        )
        if not ok:
            raise RuntimeError(f"Could not attach audio stream output: {error}")

        start_stream(self.stream, timeout=self.timeout)
        return target_description

    def read_chunk(self, duration_seconds: float) -> numpy.ndarray | None:
        if self.collector is None:
            raise RuntimeError("ContinuousAudioListener has not been started.")
        return self.collector.pop_chunk(duration_seconds)

    def stop(self) -> None:
        if self.collector is not None:
            self.collector.close()
        if self.stream is not None:
            stop_stream(self.stream, timeout=self.timeout)
            self.stream = None

    def debug_state(self) -> dict[str, object]:
        if self.collector is None:
            return {"started": False}
        return {
            "started": True,
            "callback_count": self.collector.callback_count,
            "last_chunk_samples": self.collector.last_chunk_samples,
            "last_input_sample_rate": self.collector.last_input_sample_rate,
            "last_channels": self.collector.last_channels,
            "last_error": self.collector.last_error,
        }
