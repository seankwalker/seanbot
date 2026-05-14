import unittest

from training.train_unsloth import render_messages, render_prompt


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


if __name__ == "__main__":
    unittest.main()
