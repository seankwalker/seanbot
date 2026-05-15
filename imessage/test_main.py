import json
import sqlite3
import tempfile
from pathlib import Path
import unittest

from imessage.main import (
    Message,
    TrainingPair,
    build_jsonl_messages,
    build_parser,
    build_training_pairs,
    cap_training_pairs,
    decode_attributed_body,
    fetch_messages,
    group_turns,
    is_reaction_text,
    write_jsonl,
)


def build_training_pair(text: str):
    return TrainingPair(
        chat_identifier="chat-a",
        input=text,
        output=f"{text} response",
        prompt_message_count=1,
        response_message_count=1,
        prompt_start_date=1,
        response_start_date=2,
    )


class PairingTests(unittest.TestCase):
    def test_parser_accepts_context_turns_flag(self):
        parser = build_parser()

        default_args = parser.parse_args([])
        args = parser.parse_args(["--context-turns", "2"])

        self.assertEqual(default_args.context_turns, 0)
        self.assertEqual(args.context_turns, 2)

    def test_groups_consecutive_messages_and_pairs_only_other_to_me(self):
        messages = [
            Message(1, 10, "HEY!!!", False, "chat-a", "text"),
            Message(2, 11, "are you around?", False, "chat-a", "text"),
            Message(3, 12, "yeah\nwhat's up?", True, "chat-a", "text"),
            Message(4, 13, "one more thing", True, "chat-a", "text"),
            Message(5, 14, "never mind", False, "chat-a", "text"),
        ]

        turns, skipped_empty = group_turns(messages, strip_urls=False)
        pairs, skipped_by_length = build_training_pairs(
            turns,
            min_chars=1,
            max_chars=None,
        )

        self.assertEqual(skipped_empty, 0)
        self.assertEqual(skipped_by_length, 0)
        self.assertEqual(len(turns), 3)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].input, "HEY!!!\nare you around?")
        self.assertEqual(pairs[0].output, "yeah\nwhat's up?\none more thing")

    def test_cap_training_pairs_keeps_most_recent_pairs(self):
        pairs = [
            build_training_pair("old"),
            build_training_pair("middle"),
            build_training_pair("new"),
        ]

        capped, dropped = cap_training_pairs(pairs, 2)

        self.assertEqual(dropped, 1)
        self.assertEqual([pair.input for pair in capped], ["middle", "new"])

    def test_output_length_filter_can_be_stricter_than_prompt_filter(self):
        turns = [
            # Short prompt should be kept because --min-chars still controls prompts.
            Message(1, 10, "wyd", False, "chat-a", "text"),
            Message(2, 11, "ok", True, "chat-a", "text"),
            Message(3, 12, "lol", False, "chat-a", "text"),
            Message(4, 13, "heading out soon", True, "chat-a", "text"),
        ]
        grouped_turns, _ = group_turns(turns, strip_urls=False)

        pairs, skipped_by_length = build_training_pairs(
            grouped_turns,
            min_chars=2,
            max_chars=None,
            min_output_chars=12,
        )

        self.assertEqual(skipped_by_length, 1)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].input, "lol")
        self.assertEqual(pairs[0].output, "heading out soon")

    def test_context_turns_prepend_previous_turns_to_jsonl_messages(self):
        messages = [
            Message(1, 10, "first prompt", False, "chat-a", "text"),
            Message(2, 11, "first response", True, "chat-a", "text"),
            Message(3, 12, "second prompt", False, "chat-a", "text"),
            Message(4, 13, "second response", True, "chat-a", "text"),
        ]
        turns, _ = group_turns(messages, strip_urls=False)

        pairs, skipped_by_length = build_training_pairs(
            turns,
            min_chars=1,
            max_chars=None,
            context_turns=2,
        )

        self.assertEqual(skipped_by_length, 0)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(pairs[0].context_turns), 0)
        self.assertEqual(
            [turn.text for turn in pairs[1].context_turns],
            ["first prompt", "first response"],
        )
        self.assertEqual(pairs[1].input, "second prompt")
        self.assertEqual(pairs[1].output, "second response")
        self.assertEqual(
            build_jsonl_messages(pairs[1]),
            [
                {"role": "user", "content": "first prompt"},
                {"role": "assistant", "content": "first response"},
                {"role": "user", "content": "second prompt"},
                {"role": "assistant", "content": "second response"},
            ],
        )

        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "pairs.jsonl"
            write_jsonl([pairs[1]], output_path)
            record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(record["metadata"]["context_turn_count"], 2)
        self.assertEqual(record["metadata"]["context_message_count"], 2)

    def test_context_turns_uses_fewer_previous_turns_when_available(self):
        messages = [
            Message(1, 10, "prior response", True, "chat-a", "text"),
            Message(2, 11, "current prompt", False, "chat-a", "text"),
            Message(3, 12, "target response", True, "chat-a", "text"),
        ]
        turns, _ = group_turns(messages, strip_urls=False)

        pairs, skipped_by_length = build_training_pairs(
            turns,
            min_chars=1,
            max_chars=None,
            context_turns=2,
        )

        self.assertEqual(skipped_by_length, 0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            [turn.text for turn in pairs[0].context_turns],
            ["prior response"],
        )
        self.assertEqual(
            build_jsonl_messages(pairs[0]),
            [
                {"role": "assistant", "content": "prior response"},
                {"role": "user", "content": "current prompt"},
                {"role": "assistant", "content": "target response"},
            ],
        )


class FetchMessageTests(unittest.TestCase):
    def test_identifies_reaction_text(self):
        self.assertTrue(is_reaction_text('Loved "a menace on the courts"'))
        self.assertTrue(is_reaction_text('Loved “a menace on the courts”'))
        self.assertTrue(
            is_reaction_text(
                "Loved “a menace on the courts”\nQuestioned “stop hitting on me as a kid”"
            )
        )
        self.assertFalse(is_reaction_text("loved that match yesterday"))
        self.assertFalse(is_reaction_text('Loved "a menace"\nYOU FIRST'))

    def test_decodes_attributed_body_fallback(self):
        data = b"streamtyped____NSString\x01\x94\x84\x01+hello there NSDictionary"

        self.assertEqual(decode_attributed_body(data), "hello there")

    def test_limit_fetches_most_recent_messages_then_returns_chronological(self):
        connection = sqlite3.connect(":memory:")
        try:
            cursor = connection.cursor()
            cursor.executescript(
                """
                CREATE TABLE message (
                    ROWID INTEGER PRIMARY KEY,
                    date INTEGER NOT NULL,
                    text TEXT,
                    attributedBody BLOB,
                    associated_message_type INTEGER,
                    is_from_me INTEGER NOT NULL
                );
                CREATE TABLE chat (
                    ROWID INTEGER PRIMARY KEY,
                    chat_identifier TEXT NOT NULL
                );
                CREATE TABLE chat_message_join (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL
                );
                INSERT INTO chat (ROWID, chat_identifier) VALUES (1, 'chat-a');
                INSERT INTO message (ROWID, date, text, attributedBody, associated_message_type, is_from_me) VALUES
                    (1, 100, 'oldest', NULL, 0, 0),
                    (2, 200, 'middle', NULL, 0, 1),
                    (3, 300, 'newest', NULL, 0, 0);
                INSERT INTO chat_message_join (chat_id, message_id) VALUES
                    (1, 1),
                    (1, 2),
                    (1, 3);
                """
            )

            result = fetch_messages(
                cursor,
                "chat-a",
                limit=2,
                min_date=None,
                max_date=None,
                include_reactions=False,
            )
        finally:
            connection.close()

        self.assertEqual([message.text for message in result.messages], ["middle", "newest"])

    def test_uses_attributed_body_when_text_is_null(self):
        connection = sqlite3.connect(":memory:")
        try:
            cursor = connection.cursor()
            cursor.executescript(
                """
                CREATE TABLE message (
                    ROWID INTEGER PRIMARY KEY,
                    date INTEGER NOT NULL,
                    text TEXT,
                    attributedBody BLOB,
                    associated_message_type INTEGER,
                    is_from_me INTEGER NOT NULL
                );
                CREATE TABLE chat (
                    ROWID INTEGER PRIMARY KEY,
                    chat_identifier TEXT NOT NULL
                );
                CREATE TABLE chat_message_join (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL
                );
                INSERT INTO chat (ROWID, chat_identifier) VALUES (1, 'chat-a');
                INSERT INTO message (ROWID, date, text, attributedBody, associated_message_type, is_from_me) VALUES
                    (1, 100, NULL, X'73747265616D74797065645F5F5F5F4E53537472696E67019484012B68656C6C6F207468657265204E5344696374696F6E617279', 0, 0);
                INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, 1);
                """
            )

            result = fetch_messages(
                cursor,
                "chat-a",
                limit=-1,
                min_date=None,
                max_date=None,
                include_reactions=False,
            )
        finally:
            connection.close()

        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].text, "hello there")
        self.assertEqual(result.messages[0].text_source, "attributedBody")

    def test_skips_associated_reactions_by_default(self):
        connection = sqlite3.connect(":memory:")
        try:
            cursor = connection.cursor()
            cursor.executescript(
                """
                CREATE TABLE message (
                    ROWID INTEGER PRIMARY KEY,
                    date INTEGER NOT NULL,
                    text TEXT,
                    attributedBody BLOB,
                    associated_message_type INTEGER,
                    is_from_me INTEGER NOT NULL
                );
                CREATE TABLE chat (
                    ROWID INTEGER PRIMARY KEY,
                    chat_identifier TEXT NOT NULL
                );
                CREATE TABLE chat_message_join (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL
                );
                INSERT INTO chat (ROWID, chat_identifier) VALUES (1, 'chat-a');
                INSERT INTO message (ROWID, date, text, attributedBody, associated_message_type, is_from_me) VALUES
                    (1, 100, 'Loved “hello”', NULL, 2000, 0),
                    (2, 200, 'actual message', NULL, 0, 0);
                INSERT INTO chat_message_join (chat_id, message_id) VALUES
                    (1, 1),
                    (1, 2);
                """
            )

            result = fetch_messages(
                cursor,
                "chat-a",
                limit=-1,
                min_date=None,
                max_date=None,
                include_reactions=False,
            )
        finally:
            connection.close()

        self.assertEqual([message.text for message in result.messages], ["actual message"])
        self.assertEqual(result.excluded_reaction_rows, 1)
        self.assertEqual(result.skipped_decoded_reaction_rows, 0)

    def test_skips_reaction_text_fallback(self):
        connection = sqlite3.connect(":memory:")
        try:
            cursor = connection.cursor()
            cursor.executescript(
                """
                CREATE TABLE message (
                    ROWID INTEGER PRIMARY KEY,
                    date INTEGER NOT NULL,
                    text TEXT,
                    attributedBody BLOB,
                    is_from_me INTEGER NOT NULL
                );
                CREATE TABLE chat (
                    ROWID INTEGER PRIMARY KEY,
                    chat_identifier TEXT NOT NULL
                );
                CREATE TABLE chat_message_join (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL
                );
                INSERT INTO chat (ROWID, chat_identifier) VALUES (1, 'chat-a');
                INSERT INTO message (ROWID, date, text, attributedBody, is_from_me) VALUES
                    (1, 100, 'Loved “hello”', NULL, 0),
                    (2, 200, 'actual message', NULL, 0);
                INSERT INTO chat_message_join (chat_id, message_id) VALUES
                    (1, 1),
                    (1, 2);
                """
            )

            result = fetch_messages(
                cursor,
                "chat-a",
                limit=-1,
                min_date=None,
                max_date=None,
                include_reactions=False,
            )
        finally:
            connection.close()

        self.assertEqual([message.text for message in result.messages], ["actual message"])
        self.assertEqual(result.excluded_reaction_rows, 0)
        self.assertEqual(result.skipped_decoded_reaction_rows, 1)


if __name__ == "__main__":
    unittest.main()
