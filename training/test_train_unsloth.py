import unittest

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock

from training.train_unsloth import (
    build_stopping_criteria,
    prepare_dataset,
    render_messages,
    render_prompt,
    resolve_response_end_marker,
    run_sample,
)


class FakeInputIds:
    shape = (1, 2)

    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 99

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": FakeInputIds()}
        if text == "<END>":
            return SimpleNamespace(input_ids=[42, 43])
        return SimpleNamespace(input_ids=[self.eos_token_id])

    def decode(self, token_ids, skip_special_tokens=False):
        return "good<END>ignored"


class FakeDataset:
    column_names = ["messages"]

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def select(self, indexes):
        return FakeDataset([self.rows[index] for index in indexes])

    def map(self, function, remove_columns=None, num_proc=None):
        return FakeDataset([function(row) for row in self.rows])


class PromptRenderingTests(unittest.TestCase):
    def test_render_prompt_includes_system_input_and_output(self):
        text = render_prompt("hey", "what's up", "You are Sean")

        self.assertIn("You are Sean", text)
        self.assertIn("### Statement:\nhey", text)
        self.assertIn("### Response:\nwhat's up", text)

    def test_render_prompt_appends_response_end_marker_to_outputs_only(self):
        text = render_prompt("hey", "what's up", "You are Sean", "<END>")
        sample_prompt = render_prompt("hey", "", "You are Sean", "<END>")

        self.assertIn("what's up\n<END>", text)
        self.assertNotIn("<END>", sample_prompt)

    def test_render_messages_uses_jsonl_roles_without_tokenizer_template(self):
        text = render_messages(
            [
                {"role": "user", "content": "what are you up to?"},
                {"role": "assistant", "content": "not much just chilling"},
            ],
            "You are Sean",
            "<END>",
        )

        self.assertIn("### Statement:\nwhat are you up to?", text)
        self.assertIn("### Response:\nnot much just chilling\n<END>", text)

    def test_render_messages_requires_user_and_assistant(self):
        with self.assertRaises(ValueError):
            render_messages([{"role": "user", "content": "hello"}], "You are Sean")

    def test_resolve_response_end_marker_defaults_to_tokenizer_eos(self):
        tokenizer = FakeTokenizer()

        self.assertEqual("<eos>", resolve_response_end_marker(tokenizer, None))
        self.assertEqual("<END>", resolve_response_end_marker(tokenizer, "<END>"))
        self.assertEqual("", resolve_response_end_marker(tokenizer, ""))

    def test_build_stopping_criteria_skips_tokenizer_eos(self):
        tokenizer = FakeTokenizer()

        self.assertIsNone(build_stopping_criteria(tokenizer, "<eos>"))

    def test_run_sample_stops_and_strips_response_marker(self):
        model = MagicMock()
        model.generate.return_value = [[10, 11, 12, 13, 14]]
        tokenizer = FakeTokenizer()

        fast_language_model_cls = MagicMock()
        stopping_criteria_list_cls = MagicMock(side_effect=lambda values: values)
        output = StringIO()

        with redirect_stdout(output):
            run_sample(
                model,
                tokenizer,
                "hello",
                10,
                "You are Sean",
                "<END>",
                fast_language_model_cls=fast_language_model_cls,
                stopping_criteria_list_cls=stopping_criteria_list_cls,
            )

        prompt_text = tokenizer.calls[0][0]
        self.assertIn("You are Sean", prompt_text)
        self.assertIn("### Statement:\nhello", prompt_text)
        self.assertIn("### Response:\n", prompt_text)
        self.assertNotIn("<END>", prompt_text)
        fast_language_model_cls.for_inference.assert_called_once_with(model)
        model.generate.assert_called_once()
        self.assertIn("stopping_criteria", model.generate.call_args.kwargs)
        self.assertEqual("good\n", output.getvalue())

    def test_prepare_dataset_can_limit_smoke_samples(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"hello {index}"},
                    {"role": "assistant", "content": "hey"},
                ]
            }
            for index in range(3)
        ]
        args = SimpleNamespace(
            max_train_samples=2,
            system_message="You are Sean",
            response_end_marker="",
            dataset_num_proc=1,
        )

        dataset = prepare_dataset(FakeDataset(rows), FakeTokenizer(), args)

        self.assertEqual(2, len(dataset.rows))
        self.assertIn("hello 0", dataset.rows[0]["text"])
        self.assertIn("hello 1", dataset.rows[1]["text"])


if __name__ == "__main__":
    unittest.main()
