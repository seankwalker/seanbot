import unittest

from unittest.mock import MagicMock

from training.train_unsloth import render_messages, render_prompt, run_sample


class PromptRenderingTests(unittest.TestCase):
    def test_render_prompt_includes_system_input_and_output(self):
        text = render_prompt("hey", "what's up", "You are Sean")

        self.assertIn("You are Sean", text)
        self.assertIn("### Statement:\nhey", text)
        self.assertIn("### Response:\nwhat's up", text)

    def test_render_messages_uses_jsonl_roles_without_tokenizer_template(self):
        text = render_messages(
            [
                {"role": "user", "content": "what are you up to?"},
                {"role": "assistant", "content": "not much just chilling"},
            ],
            "You are Sean",
        )

        self.assertIn("### Statement:\nwhat are you up to?", text)
        self.assertIn("### Response:\nnot much just chilling", text)

    def test_render_messages_requires_user_and_assistant(self):
        with self.assertRaises(ValueError):
            render_messages([{"role": "user", "content": "hello"}], "You are Sean")

    def test_run_sample_tokenizes_rendered_prompt_without_chat_template(self):
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.return_value.input_ids.to.return_value = "cuda-input-ids"
        tokenizer.eos_token_id = 128001

        text_streamer_cls = MagicMock()
        fast_language_model_cls = MagicMock()
        run_sample(
            model,
            tokenizer,
            "hello",
            10,
            "You are Sean",
            text_streamer_cls=text_streamer_cls,
            fast_language_model_cls=fast_language_model_cls,
        )

        tokenizer.assert_called_once()
        prompt_text = tokenizer.call_args.args[0]
        self.assertIn("You are Sean", prompt_text)
        self.assertIn("### Statement:\nhello", prompt_text)
        self.assertIn("### Response:\n", prompt_text)
        fast_language_model_cls.for_inference.assert_called_once_with(model)
        text_streamer_cls.assert_called_once_with(tokenizer, skip_prompt=True)
        model.generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
