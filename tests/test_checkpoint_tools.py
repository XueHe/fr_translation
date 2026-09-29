from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


inspect_checkpoint = load_module(
    "inspect_checkpoint", ROOT / "tools" / "inspect_checkpoint.py"
)
rollback_journal = load_module(
    "rollback_journal", ROOT / "tools" / "rollback_journal.py"
)


class CheckpointToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.journal = Path(self.temp.name) / "raw_responses.jsonl"
        envelopes = []
        for batch_index, start in enumerate((0, 2, 4)):
            envelopes.append(
                {
                    "schema_version": 1,
                    "shard": 0,
                    "batch_index": batch_index,
                    "input_start": start,
                    "input_end": start + 2,
                    "batch_seconds": 1.5,
                    "records": [{"term_id": f"t{start}"}, {"term_id": f"t{start + 1}"}],
                }
            )
        self.journal.write_text(
            "".join(json.dumps(item) + "\n" for item in envelopes), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_inspect_reports_contiguous_offset(self) -> None:
        offset, batches, seconds = inspect_checkpoint.inspect_journal(self.journal, 0)
        self.assertEqual((offset, batches, seconds), (6, 3, 4.5))

    def test_rollback_finds_exact_boundary(self) -> None:
        boundary, current, lines = rollback_journal.locate_boundary(self.journal, 0, 4)
        self.assertEqual((current, lines), (6, 3))
        self.assertGreater(boundary, 0)

    def test_rollback_rejects_non_boundary(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not a batch boundary"):
            rollback_journal.locate_boundary(self.journal, 0, 3)


if __name__ == "__main__":
    unittest.main()
