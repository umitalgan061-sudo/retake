"""Regression tests for Windows symlink preflight (no Hugging Face download)."""
import io
import inspect
import sys
import types
import unittest
from unittest.mock import patch

from tools import retake_realism_worker as worker


class HuggingFaceCachePreflightTest(unittest.TestCase):
    def test_worker_probes_cache_before_parallel_model_loading(self):
        source = inspect.getsource(worker.main)
        self.assertLess(
            source.index("prepare_windows_huggingface_cache()"),
            source.index("renderer = Renderer("),
        )

    def test_non_windows_leaves_huggingface_untouched(self):
        with patch.object(worker.sys, "platform", "linux"):
            worker.prepare_windows_huggingface_cache()

    def test_windows_checks_symlinks_before_download_threads(self):
        calls = []
        pkg = types.ModuleType("huggingface_hub")
        pkg.__path__ = []
        file_download = types.ModuleType("huggingface_hub.file_download")
        file_download.are_symlinks_supported = lambda: calls.append("checked") or False

        with patch.dict(sys.modules, {
            "huggingface_hub": pkg,
            "huggingface_hub.file_download": file_download,
        }):
            with patch.object(worker.sys, "platform", "win32"):
                with patch.object(worker.sys, "stderr", new=io.StringIO()) as output:
                    worker.prepare_windows_huggingface_cache()
                    self.assertIn("WinError 1314", output.getvalue())
        self.assertEqual(calls, ["checked"])

    def test_windows_keeps_symlink_enabled_when_permitted(self):
        calls = []
        pkg = types.ModuleType("huggingface_hub")
        pkg.__path__ = []
        file_download = types.ModuleType("huggingface_hub.file_download")
        file_download.are_symlinks_supported = lambda: calls.append("checked") or True
        with patch.dict(sys.modules, {
            "huggingface_hub": pkg,
            "huggingface_hub.file_download": file_download,
        }):
            with patch.object(worker.sys, "platform", "win32"):
                with patch.object(worker.sys, "stderr", new=io.StringIO()) as output:
                    worker.prepare_windows_huggingface_cache()
                    self.assertEqual(output.getvalue(), "")
        self.assertEqual(calls, ["checked"])


if __name__ == "__main__":
    unittest.main()
