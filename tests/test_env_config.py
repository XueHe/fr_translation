from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("translation", ROOT / "scripts/translation_llama_full.py")
translation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(translation)


class EnvConfigTest(unittest.TestCase):
    def test_load_paths_and_preserve_scheduler_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / ".env"
            config.write_text('OUTPUT_DIR="/private/path with spaces"\nMODEL_DIR=/models/llama\n')
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-one,GPU-two,GPU-three,GPU-four"}):
                translation.load_env_file(config)
                self.assertEqual(os.environ["OUTPUT_DIR"], "/private/path with spaces")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-one,GPU-two,GPU-three,GPU-four")

    def test_explicit_missing_config_fails(self):
        with self.assertRaises(FileNotFoundError):
            translation.load_env_file(Path("/nonexistent/fr_translation.env"), required=True)

    def test_status_uses_env_without_model_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / ".env"
            config.write_text(f'OUTPUT_DIR="{directory}"\n')
            with patch.dict(os.environ), patch("sys.argv", ["translation", "--env-file", str(config), "status"]):
                translation.main()


if __name__ == "__main__":
    unittest.main()
