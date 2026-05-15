# iMessage Data Extraction

This module extracts local iMessage conversations into prompt/response training
pairs for supervised fine-tuning. The goal is to preserve tone and style while
building cleaner examples than raw message dumps.

## Requirements

- macOS with access to `~/Library/Messages/chat.db`
- Python 3.12
- `uv` and `mise` if using the pinned runtime in `mise.toml`

The script opens the Messages database in read-only mode.
Recent macOS versions often store message bodies in `attributedBody`; the
project uses `pytypedstream` to decode those rows.

## Basic Usage

Run from this directory:

```bash
uv run main.py \
  --chat-ids "+15551234567,+15557654321" \
  --limit 1000 \
  --output training_pairs.csv \
  --jsonl-output training_pairs.jsonl
```

`--limit` reads the most recent text-bearing messages per chat, then sorts them
chronologically before building pairs. Rows with `message.text` or decodable
`message.attributedBody` are considered text-bearing.
iMessage tapbacks/reactions are skipped by default.

## Chat ID Files

For longer chat lists, use one identifier per line:

```text
+15551234567
+15557654321
# Comments are ignored
```

Then run:

```bash
uv run main.py \
  --chat-ids-file chat_ids.txt \
  --min-date 2024-01-01 \
  --output training_pairs.csv
```

## Useful Options

- `--db-path`: override the default `~/Library/Messages/chat.db` path.
- `--min-date` / `--max-date`: filter by local date or ISO datetime.
- `--min-chars` / `--max-chars`: drop turns outside a length range.
- `--min-output-chars` / `--max-output-chars`: apply a separate length range
  to your response turns. Use this to keep short prompts while dropping terse
  training targets.
- `--max-pairs-per-chat`: cap each chat's contribution after filtering. The
  most recent usable pairs are kept.
- `--context-turns`: prepend up to this many previous turns to each JSONL
  record. CSV output stays as the current prompt/response only.
- `--strip-urls`: remove URLs while preserving the rest of the message style.
- `--include-reactions`: keep tapbacks/reactions such as `Loved “...”`.
- `--jsonl-output`: write chat-style JSONL records for SFT workflows.

## Balancing Chats

Use `--max-pairs-per-chat` when one long conversation dominates the dataset:

```bash
uv run main.py \
  --chat-ids-file chat_ids.txt \
  --limit 15000 \
  --max-pairs-per-chat 1500 \
  --output training_pairs.csv \
  --jsonl-output training_pairs.jsonl
```

Group chats are supported by `chat_identifier`, but all non-you messages are
grouped as the prompt side. For cleaner tone modeling, start with 1:1 chats and
add group chats only after manually reviewing samples.

For experiments that should reduce low-information replies, keep prompt
filtering permissive and raise only the output threshold:

```bash
uv run main.py \
  --chat-ids-file chat_ids.txt \
  --limit 5000 \
  --max-pairs-per-chat 300 \
  --min-chars 2 \
  --min-output-chars 12 \
  --jsonl-output exp-b-balanced-min-response.jsonl
```

Experiment C adds short conversation context while keeping the same final
assistant response as the training target:

```bash
uv run main.py \
  --chat-ids-file chat_ids.txt \
  --context-turns 2 \
  --min-output-chars 12 \
  --limit 5000 \
  --max-pairs-per-chat 300 \
  --jsonl-output exp-c-context-turns.jsonl
```

## Output Formats

CSV output contains:

```csv
input,output
```

JSONL output contains one record per pair:

```json
{"messages":[{"role":"user","content":"previous prompt"},{"role":"assistant","content":"previous response"},{"role":"user","content":"current prompt"},{"role":"assistant","content":"target response"}],"metadata":{"context_turn_count":2}}
```

When `--context-turns` is omitted, JSONL keeps the existing one user prompt
turn plus one assistant response turn shape. The final assistant message remains
the training target.

## Validation

Run the unit tests after changing extraction or pairing logic:

```bash
python -m unittest imessage.test_main
```

## Privacy

Generated CSV and JSONL files may contain personal messages, phone numbers,
emails, links, and other PII. Do not commit raw exports or training outputs
unless they have been intentionally sanitized.
