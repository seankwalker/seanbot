import json
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from training.eval_checkpoint import (
    build_metadata,
    build_record,
    prepare_output_path,
    read_prompts,
)


class EvalCheckpointTests(unittest.TestCase):
    def test_read_prompts_ignores_comments_and_blank_lines(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "prompts.txt"
            path.write_text(
                "# comment\n\nhello\n  how are you?  \n",
                encoding="utf-8",
            )

            self.assertEqual(["hello", "how are you?"], read_prompts(path))

    def test_prepare_output_path_requires_overwrite_for_existing_file(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "eval.jsonl"
            path.write_text("old\n", encoding="utf-8")

            with self.assertRaises(SystemExit):
                prepare_output_path(path, overwrite=False)

            prepare_output_path(path, overwrite=True)
            self.assertEqual("", path.read_text(encoding="utf-8"))

    def test_build_record_is_jsonl_serializable(self):
        args = SimpleNamespace(
            run_label="smoke",
            model_name="checkpoint-60",
            prompts=Path("training/eval_prompts.txt"),
            samples_per_prompt=2,
            sample_max_new_tokens=40,
            do_sample=True,
            temperature=0.65,
            top_p=0.9,
            seed=3407,
            system_message="You are Sean",
            response_end_marker=None,
        )
        metadata = build_metadata(args, ["hello"], "2026-05-14T00:00:00+00:00")
        record = build_record(metadata, 0, 1, "hello", "hi")

        encoded = json.dumps(record)
        self.assertIn('"prompt": "hello"', encoded)
        self.assertEqual("hi", record["response"])
        self.assertEqual("smoke", record["metadata"]["run_label"])


if __name__ == "__main__":
    unittest.main()
