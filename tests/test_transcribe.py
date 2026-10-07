"""Offline regression tests: no model downloads or GPU are required."""

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import transcribe

TEST_TEMP_ROOT = Path(__file__).resolve().parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class SourceValidationTests(unittest.TestCase):
    def test_supported_url_forms_point_to_the_same_full_video(self):
        video_id = "AbC_dE-1234"
        expected = f"https://www.youtube.com/watch?v={video_id}"
        for url in (
            f"https://youtu.be/{video_id}?t=10",
            f"https://www.youtube.com/watch?v={video_id}&list=playlist",
            f"https://m.youtube.com/shorts/{video_id}",
            f"https://youtube.com/live/{video_id}",
            f"https://youtube.com/embed/{video_id}",
        ):
            with self.subTest(url=url):
                self.assertEqual(transcribe.normalize_youtube_url(url), expected)

    def test_other_hosts_and_playlists_are_rejected(self):
        for url in (
            "https://youtube.com.evil.example/watch?v=AbC_dE-1234",
            "https://youtube.com/playlist?list=123",
            "https://youtu.be/invalid",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                transcribe.normalize_youtube_url(url)

    def test_windows_paths_and_reserved_filenames(self):
        self.assertIsNone(transcribe.normalize_youtube_url(r"C:\Media\lecture.m4a"))
        for name in ("CON", "nul.txt", "LPT1", "COM9"):
            self.assertTrue(transcribe.safe_filename(name).startswith("transcript_"))
        self.assertEqual(transcribe.safe_filename("lecture: part/one?"), "lecture_ part_one_")

    def test_list_file_supports_bom_comments_and_spaces(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as directory:
            source_list = Path(directory) / "urls.txt"
            source_list.write_text("# comment\n\n C:/My Media/lecture.m4a \n", encoding="utf-8-sig")
            options = argparse.Namespace(sources=["first.wav"], list=source_list, diagnose=False)
            self.assertEqual(
                transcribe.collect_sources(options), ["first.wav", "C:/My Media/lecture.m4a"]
            )


class TranscriptLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.media = self.root / "lecture.m4a"
        self.media.write_bytes(b"fake audio; decoding is handled by the test engine")
        self.options = argparse.Namespace(
            language="es",
            batch_size=1,
            model="small",
            output_dir=self.root / "output",
            no_timestamps=False,
        )
        self.audio_info = SimpleNamespace(language="es", duration=65.0)
        self.segment = SimpleNamespace(start=1.0, end=4.0, text="  Hello, class — café.  ")
        self.output = io.StringIO()

    def engine(self, segments):
        model = MagicMock()
        model.transcribe.return_value = (segments, self.audio_info)
        return transcribe.TranscriptionEngine(model, "cpu", "int8", None)

    def test_completed_output_is_utf8_with_an_english_report(self):
        with redirect_stdout(self.output):
            report = transcribe.transcribe_source(
                str(self.media), self.engine(iter([self.segment])), self.options
            )
        transcript = Path(report["transcript_path"])
        self.assertIn(
            "[00:00:01 - 00:00:04] Hello, class — café.", transcript.read_text(encoding="utf-8")
        )
        self.assertEqual(
            json.loads(transcript.with_suffix(".json").read_text(encoding="utf-8"))["device"], "cpu"
        )
        self.assertFalse(list(self.options.output_dir.glob("*.partial")))

    def test_failure_preserves_partial_text_without_publishing_a_final_txt(self):
        def interrupted_segments():
            yield self.segment
            raise RuntimeError("decoder failed")

        with (
            redirect_stdout(self.output),
            redirect_stderr(self.output),
            self.assertRaisesRegex(RuntimeError, "decoder failed"),
        ):
            transcribe.transcribe_source(
                str(self.media), self.engine(interrupted_segments()), self.options
            )
        self.assertFalse(list(self.options.output_dir.glob("*.txt")))
        partial_files = list(self.options.output_dir.glob("*.txt.partial"))
        self.assertEqual(len(partial_files), 1)
        self.assertIn("Hello, class — café.", partial_files[0].read_text(encoding="utf-8"))

    def test_repeated_runs_do_not_overwrite_existing_transcripts(self):
        with redirect_stdout(self.output):
            first = transcribe.transcribe_source(
                str(self.media), self.engine(iter([self.segment])), self.options
            )
            second = transcribe.transcribe_source(
                str(self.media), self.engine(iter([self.segment])), self.options
            )
        self.assertNotEqual(first["transcript_path"], second["transcript_path"])
        self.assertEqual(len(list(self.options.output_dir.glob("*.txt"))), 2)


class DeviceSelectionTests(unittest.TestCase):
    def test_explicit_cpu_mode_does_not_probe_cuda(self):
        translator = SimpleNamespace(
            get_cuda_device_count=MagicMock(side_effect=RuntimeError("broken driver"))
        )
        model_factory = MagicMock()
        whisper = SimpleNamespace(WhisperModel=model_factory, BatchedInferencePipeline=MagicMock())
        options = argparse.Namespace(device="cpu", model="small", batch_size=1, cpu_threads=4)
        with (
            patch.dict("sys.modules", {"ctranslate2": translator, "faster_whisper": whisper}),
            redirect_stdout(io.StringIO()),
        ):
            engine = transcribe.load_engine(options)
        translator.get_cuda_device_count.assert_not_called()
        self.assertEqual(engine.device, "cpu")
        self.assertEqual(engine.compute_type, "int8")

    def test_missing_cuda_libraries_do_not_trigger_a_silent_cpu_retry(self):
        translator = SimpleNamespace(get_cuda_device_count=lambda: 1)
        model_factory = MagicMock(side_effect=RuntimeError("missing CUDA library"))
        whisper = SimpleNamespace(WhisperModel=model_factory, BatchedInferencePipeline=MagicMock())
        options = argparse.Namespace(device="cuda", model="small", batch_size=1, cpu_threads=4)
        with (
            patch.dict("sys.modules", {"ctranslate2": translator, "faster_whisper": whisper}),
            patch("transcribe.configure_cuda_libraries"),
            patch("transcribe.query_nvidia_gpu", return_value="GPU"),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "missing CUDA library"),
        ):
            transcribe.load_engine(options)
        self.assertEqual(model_factory.call_count, 1)
        self.assertEqual(model_factory.call_args.kwargs["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
