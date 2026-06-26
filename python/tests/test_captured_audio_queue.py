from __future__ import annotations

"""Contract tests for captured audio queue filesystem lifecycle.

These tests define the expected producer/consumer behavior for the future
`modules.captured_audio_queue` module:
- publish finished audio into a ready state under `inbox`
- claim the oldest ready file into `processing`
- move processed files into `done` or `failed`
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.captured_audio_queue import (
    claim_oldest_ready_audio,
    mark_audio_processing_failed,
    mark_audio_processing_succeeded,
    parse_conversation_id_from_captured_audio_filename,
    publish_captured_audio,
)


class CapturedAudioQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name)
        self.audio = numpy.array([0.0, 0.25, -0.25, 0.5], dtype=numpy.float32)
        self.sample_rate = 16000

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_publish_creates_expected_directories_and_places_finished_file_in_inbox(self) -> None:
        # Publishing is the producer boundary: the caller hands over finished audio,
        # and the queue module materializes the storage layout and exposes only a ready file in inbox.
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=self.sample_rate,
            storage_dir=self.storage_dir,
            conversation_id=17,
            prefix="utterance",
        )

        # The observable contract is filesystem state, not internal write steps.
        # A published file must already be finalized, discoverable in inbox, and leave no partial temp artifact behind.
        self.assertEqual(self.storage_dir / "inbox", published_path.parent)
        self.assertTrue(published_path.exists())
        self.assertEqual(".wav", published_path.suffix)
        self.assertFalse(any((self.storage_dir / ".tmp").glob("*.partial.wav")))
        self.assertTrue((self.storage_dir / "processing").is_dir())
        self.assertTrue((self.storage_dir / "done").is_dir())
        self.assertTrue((self.storage_dir / "failed").is_dir())
        self.assertIn("utterance-conv-17-", published_path.name)

    def test_claim_oldest_ready_audio_moves_the_oldest_inbox_file_to_processing(self) -> None:
        first_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=self.sample_rate,
            storage_dir=self.storage_dir,
            conversation_id=17,
            prefix="utterance",
        )
        second_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=self.sample_rate,
            storage_dir=self.storage_dir,
            conversation_id=17,
            prefix="utterance",
        )

        claimed_path = claim_oldest_ready_audio(self.storage_dir)

        self.assertEqual(first_path.name, claimed_path.name)
        self.assertEqual(self.storage_dir / "processing", claimed_path.parent)
        self.assertFalse(first_path.exists())
        self.assertTrue(claimed_path.exists())
        self.assertTrue((self.storage_dir / "inbox" / second_path.name).exists())

    def test_claim_returns_none_when_inbox_is_empty(self) -> None:
        self.assertIsNone(claim_oldest_ready_audio(self.storage_dir))

    def test_parse_conversation_id_from_captured_audio_filename_returns_embedded_id(self) -> None:
        self.assertEqual(
            17,
            parse_conversation_id_from_captured_audio_filename(
                "utterance-conv-17-20260423-123456-654321.wav"
            ),
        )

    def test_parse_conversation_id_from_captured_audio_filename_rejects_invalid_format(self) -> None:
        with self.assertRaises(ValueError):
            parse_conversation_id_from_captured_audio_filename("utterance-20260423-123456-654321.wav")

    def test_claim_skips_file_that_was_taken_by_another_worker_before_replace(self) -> None:
        first_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=self.sample_rate,
            storage_dir=self.storage_dir,
            conversation_id=17,
            prefix="utterance",
        )
        second_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=self.sample_rate,
            storage_dir=self.storage_dir,
            conversation_id=17,
            prefix="utterance",
        )
        first_inbox_path = self.storage_dir / "inbox" / first_path.name
        original_replace = Path.replace
        state = {"failed_once": False}

        def replace_with_race(path: Path, target: Path) -> Path:
            if path == first_inbox_path and not state["failed_once"]:
                state["failed_once"] = True
                first_inbox_path.unlink()
                raise FileNotFoundError(str(path))
            return original_replace(path, target)

        with patch.object(Path, "replace", new=replace_with_race):
            claimed_path = claim_oldest_ready_audio(self.storage_dir)

        self.assertEqual(second_path.name, claimed_path.name)
        self.assertEqual(self.storage_dir / "processing", claimed_path.parent)
        self.assertTrue(claimed_path.exists())

    def test_mark_audio_processing_succeeded_moves_processing_file_to_done(self) -> None:
        publish_captured_audio(
            audio=self.audio,
            sample_rate=self.sample_rate,
            storage_dir=self.storage_dir,
            conversation_id=17,
            prefix="utterance",
        )
        processing_path = claim_oldest_ready_audio(self.storage_dir)

        done_path = mark_audio_processing_succeeded(processing_path)

        self.assertEqual(self.storage_dir / "done", done_path.parent)
        self.assertTrue(done_path.exists())
        self.assertFalse(processing_path.exists())

    def test_mark_audio_processing_failed_moves_processing_file_to_failed(self) -> None:
        publish_captured_audio(
            audio=self.audio,
            sample_rate=self.sample_rate,
            storage_dir=self.storage_dir,
            conversation_id=17,
            prefix="utterance",
        )
        processing_path = claim_oldest_ready_audio(self.storage_dir)

        failed_path = mark_audio_processing_failed(processing_path)

        self.assertEqual(self.storage_dir / "failed", failed_path.parent)
        self.assertTrue(failed_path.exists())
        self.assertFalse(processing_path.exists())

    def test_mark_succeeded_rejects_non_processing_paths(self) -> None:
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=self.sample_rate,
            storage_dir=self.storage_dir,
            conversation_id=17,
            prefix="utterance",
        )

        with self.assertRaises(ValueError):
            mark_audio_processing_succeeded(published_path)

    def test_mark_failed_rejects_non_processing_paths(self) -> None:
        published_path = publish_captured_audio(
            audio=self.audio,
            sample_rate=self.sample_rate,
            storage_dir=self.storage_dir,
            conversation_id=17,
            prefix="utterance",
        )

        with self.assertRaises(ValueError):
            mark_audio_processing_failed(published_path)


if __name__ == "__main__":
    unittest.main()
