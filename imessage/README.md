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
- `--max-pairs-per-chat`: cap each chat's contribution after filtering. The
  most recent usable pairs are kept.
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

## Output Formats

CSV output contains:

```csv
input,output
```

JSONL output contains one record per pair:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Validation

Run the unit tests after changing extraction or pairing logic:

```bash
python -m unittest imessage.test_main
```

## Privacy

Generated CSV and JSONL files may contain personal messages, phone numbers,
emails, links, and other PII. Do not commit raw exports or training outputs
unless they have been intentionally sanitized.
