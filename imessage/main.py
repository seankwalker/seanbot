import argparse
import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from urllib.parse import quote


APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
APPLE_NANOSECONDS_PER_SECOND = 1_000_000_000
REACTION_LINE_PATTERN = re.compile(
    r"^(?:"
    r"(?:Liked|Loved|Disliked|Laughed at|Emphasized|Questioned)\s+[\"“].+[\"”]"
    r"|Removed\s+(?:a|an)\s+.+\s+from\s+[\"“].+[\"”]"
    r"|Reacted(?:\s+with\s+.+)?\s+to\s+[\"“].+[\"”]"
    r")$"
)


@dataclass(frozen=True)
class Message:
    message_id: int
    date: int
    text: str
    is_from_me: bool
    chat_identifier: str
    text_source: str


@dataclass(frozen=True)
class FetchResult:
    messages: list[Message]
    raw_rows: int
    candidate_rows: int
    text_rows: int
    attributed_body_rows: int
    rows_fetched: int
    decoded_attributed_rows: int
    excluded_reaction_rows: int
    skipped_decoded_reaction_rows: int
    skipped_undecoded_rows: int


@dataclass(frozen=True)
class Turn:
    chat_identifier: str
    is_from_me: bool
    text: str
    message_count: int
    start_date: int
    end_date: int


@dataclass(frozen=True)
class TrainingPair:
    chat_identifier: str
    input: str
    output: str
    prompt_message_count: int
    response_message_count: int
    prompt_start_date: int
    response_start_date: int
    context_turns: tuple[Turn, ...] = ()


def parse_apple_timestamp(value: str, *, end_of_day: bool = False) -> int:
    local_tz = datetime.now().astimezone().tzinfo
    raw_value = value.strip()

    if len(raw_value) == 10 and raw_value[4] == "-" and raw_value[7] == "-":
        date_value = datetime.fromisoformat(raw_value).date()
        boundary = time.max if end_of_day else time.min
        dt = datetime.combine(date_value, boundary, tzinfo=local_tz)
    else:
        dt = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=local_tz)

    delta = dt.astimezone(timezone.utc) - APPLE_EPOCH
    total_microseconds = (
        ((delta.days * 24 * 60 * 60) + delta.seconds) * 1_000_000
    ) + delta.microseconds
    return total_microseconds * 1_000


def normalize_text(text: str, *, strip_urls: bool) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if strip_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text).strip()
    return text


def is_reaction_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(REACTION_LINE_PATTERN.match(line) for line in lines)


def decode_attributed_body(data: bytes | None) -> str | None:
    if not data:
        return None

    if isinstance(data, memoryview):
        data = data.tobytes()

    typedstream_text = decode_attributed_body_with_typedstream(data)
    if typedstream_text:
        return typedstream_text

    return decode_attributed_body_fallback(data)


def decode_attributed_body_with_typedstream(data: bytes) -> str | None:
    try:
        from typedstream.stream import TypedStreamReader
    except ImportError:
        return None

    try:
        for event in TypedStreamReader.from_data(data):
            if isinstance(event, bytes):
                text = event.decode("utf-8", errors="replace").strip()
                if text:
                    return text
    except Exception:
        return None

    return None


def decode_attributed_body_fallback(data: bytes) -> str | None:
    decoded = data.decode("utf-8", errors="ignore")
    start_markers = ("NSString", "NSMutableString")
    stop_markers = (
        "NSDictionary",
        "NSMutableDictionary",
        "NSNumber",
        "NSValue",
        "__kIM",
    )

    for start_marker in start_markers:
        start = decoded.find(start_marker)
        if start == -1:
            continue

        tail = decoded[start + len(start_marker) :]
        stop_positions = [
            tail.find(stop_marker)
            for stop_marker in stop_markers
            if tail.find(stop_marker) != -1
        ]
        if stop_positions:
            tail = tail[: min(stop_positions)]

        text = "".join(char for char in tail if char.isprintable() or char in "\n\t")
        text = text.strip()
        if text.startswith("+") and len(text) > 1 and not text[1].isdigit():
            text = text[1:].lstrip()
        if text:
            return text

    return None


def read_chat_ids(chat_ids: str | None, chat_ids_file: Path | None) -> list[str]:
    values: list[str] = []

    if chat_ids:
        values.extend(chat_id.strip() for chat_id in chat_ids.split(","))

    if chat_ids_file:
        with chat_ids_file.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    values.append(line)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)

    return deduped


def get_message_columns(cursor: sqlite3.Cursor) -> set[str]:
    return {row[1] for row in cursor.execute("PRAGMA table_info(message)").fetchall()}


def fetch_messages(
    cursor: sqlite3.Cursor,
    chat_identifier: str,
    *,
    limit: int,
    min_date: int | None,
    max_date: int | None,
    include_reactions: bool,
) -> FetchResult:
    base_where_clauses = [
        "chat.chat_identifier = ?",
    ]
    base_params: list[object] = [chat_identifier]

    if min_date is not None:
        base_where_clauses.append("message.date >= ?")
        base_params.append(min_date)
    if max_date is not None:
        base_where_clauses.append("message.date <= ?")
        base_params.append(max_date)

    message_columns = get_message_columns(cursor)
    reaction_filter_sql = None
    if not include_reactions and "associated_message_type" in message_columns:
        reaction_filter_sql = (
            "(message.associated_message_type IS NULL "
            "OR message.associated_message_type = 0)"
        )

    raw_where_sql = " AND ".join(base_where_clauses)
    raw_count_query = f"""
        SELECT COUNT(*)
        FROM message
        JOIN chat_message_join ON chat_message_join.message_id = message.ROWID
        JOIN chat ON chat.ROWID = chat_message_join.chat_id
        WHERE {raw_where_sql}
    """
    (raw_rows,) = cursor.execute(raw_count_query, base_params).fetchone()

    excluded_reaction_rows = 0
    if reaction_filter_sql is not None:
        reaction_count_query = f"""
            SELECT SUM(CASE WHEN NOT ({reaction_filter_sql}) THEN 1 ELSE 0 END)
            FROM message
            JOIN chat_message_join ON chat_message_join.message_id = message.ROWID
            JOIN chat ON chat.ROWID = chat_message_join.chat_id
            WHERE {raw_where_sql}
        """
        (excluded_reaction_rows,) = cursor.execute(
            reaction_count_query,
            base_params,
        ).fetchone()

    where_clauses = list(base_where_clauses)
    if reaction_filter_sql is not None:
        where_clauses.append(reaction_filter_sql)
    where_sql = " AND ".join(where_clauses)
    count_query = f"""
        SELECT
            SUM(CASE WHEN message.text IS NOT NULL AND message.text != '' THEN 1 ELSE 0 END) AS text_rows,
            SUM(CASE WHEN message.attributedBody IS NOT NULL THEN 1 ELSE 0 END) AS attributed_body_rows,
            SUM(
                CASE
                    WHEN (message.text IS NOT NULL AND message.text != '')
                      OR message.attributedBody IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) AS candidate_rows
        FROM message
        JOIN chat_message_join ON chat_message_join.message_id = message.ROWID
        JOIN chat ON chat.ROWID = chat_message_join.chat_id
        WHERE {where_sql}
    """
    text_rows, attributed_body_rows, candidate_rows = cursor.execute(
        count_query,
        base_params,
    ).fetchone()

    candidate_where_sql = (
        f"{where_sql} AND ("
        "(message.text IS NOT NULL AND message.text != '') "
        "OR message.attributedBody IS NOT NULL)"
    )
    base_query = f"""
        SELECT
            message.ROWID AS message_id,
            message.date AS message_date,
            message.text,
            message.attributedBody,
            message.is_from_me,
            chat.chat_identifier
        FROM message
        JOIN chat_message_join ON chat_message_join.message_id = message.ROWID
        JOIN chat ON chat.ROWID = chat_message_join.chat_id
        WHERE {candidate_where_sql}
    """

    if limit == -1:
        query = f"{base_query} ORDER BY message.date ASC, message.ROWID ASC"
    else:
        query = f"""
            SELECT * FROM (
                {base_query}
                ORDER BY message.date DESC, message.ROWID DESC
                LIMIT ?
            )
            ORDER BY message_date ASC, message_id ASC
        """

    query_params = list(base_params)
    if limit != -1:
        query_params.append(limit)

    rows = cursor.execute(query, query_params).fetchall()
    messages: list[Message] = []
    decoded_attributed_rows = 0
    skipped_decoded_reaction_rows = 0
    skipped_undecoded_rows = 0

    for row in rows:
        text = row[2]
        text_source = "text"

        if not text:
            text = decode_attributed_body(row[3])
            text_source = "attributedBody"
            if text:
                decoded_attributed_rows += 1
            else:
                skipped_undecoded_rows += 1
                continue

        if not include_reactions and is_reaction_text(text):
            skipped_decoded_reaction_rows += 1
            continue

        messages.append(
            Message(
                message_id=row[0],
                date=row[1],
                text=text,
                is_from_me=bool(row[4]),
                chat_identifier=row[5],
                text_source=text_source,
            )
        )

    return FetchResult(
        messages=messages,
        raw_rows=raw_rows or 0,
        candidate_rows=candidate_rows or 0,
        text_rows=text_rows or 0,
        attributed_body_rows=attributed_body_rows or 0,
        rows_fetched=len(rows),
        decoded_attributed_rows=decoded_attributed_rows,
        excluded_reaction_rows=excluded_reaction_rows or 0,
        skipped_decoded_reaction_rows=skipped_decoded_reaction_rows,
        skipped_undecoded_rows=skipped_undecoded_rows,
    )


def group_turns(
    messages: list[Message],
    *,
    strip_urls: bool,
    message_separator: str = "\n",
) -> tuple[list[Turn], int]:
    turns: list[Turn] = []
    skipped_empty = 0
    current_messages: list[Message] = []
    current_texts: list[str] = []

    def flush_current() -> None:
        if not current_messages:
            return
        turns.append(
            Turn(
                chat_identifier=current_messages[0].chat_identifier,
                is_from_me=current_messages[0].is_from_me,
                text=message_separator.join(current_texts),
                message_count=len(current_messages),
                start_date=current_messages[0].date,
                end_date=current_messages[-1].date,
            )
        )

    for message in messages:
        text = normalize_text(message.text, strip_urls=strip_urls)
        if not text:
            skipped_empty += 1
            continue

        if current_messages and current_messages[-1].is_from_me != message.is_from_me:
            flush_current()
            current_messages = []
            current_texts = []

        current_messages.append(message)
        current_texts.append(text)

    flush_current()
    return turns, skipped_empty


def build_training_pairs(
    turns: list[Turn],
    *,
    min_chars: int,
    max_chars: int | None,
    min_output_chars: int | None = None,
    max_output_chars: int | None = None,
    context_turns: int = 0,
) -> tuple[list[TrainingPair], int]:
    if context_turns < 0:
        raise ValueError("context_turns must be 0 or greater.")

    pairs: list[TrainingPair] = []
    skipped_by_length = 0
    effective_min_output_chars = (
        min_chars if min_output_chars is None else min_output_chars
    )
    effective_max_output_chars = (
        max_chars if max_output_chars is None else max_output_chars
    )

    for index, (prompt, response) in enumerate(zip(turns, turns[1:])):
        if prompt.is_from_me or not response.is_from_me:
            continue

        if len(prompt.text) < min_chars or len(response.text) < effective_min_output_chars:
            skipped_by_length += 1
            continue
        if max_chars is not None and len(prompt.text) > max_chars:
            skipped_by_length += 1
            continue
        if (
            effective_max_output_chars is not None
            and len(response.text) > effective_max_output_chars
        ):
            skipped_by_length += 1
            continue

        context_start = max(0, index - context_turns)
        previous_turns = turns[context_start:index] if context_turns else []

        pairs.append(
            TrainingPair(
                chat_identifier=prompt.chat_identifier,
                input=prompt.text,
                output=response.text,
                prompt_message_count=prompt.message_count,
                response_message_count=response.message_count,
                prompt_start_date=prompt.start_date,
                response_start_date=response.start_date,
                context_turns=tuple(previous_turns),
            )
        )

    return pairs, skipped_by_length


def cap_training_pairs(
    pairs: list[TrainingPair],
    max_pairs: int | None,
) -> tuple[list[TrainingPair], int]:
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs, 0

    return pairs[-max_pairs:], len(pairs) - max_pairs


def write_csv(pairs: list[TrainingPair], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["input", "output"])
        for pair in pairs:
            writer.writerow([pair.input, pair.output])


def turn_to_chat_message(turn: Turn) -> dict[str, str]:
    role = "assistant" if turn.is_from_me else "user"
    return {"role": role, "content": turn.text}


def build_jsonl_messages(pair: TrainingPair) -> list[dict[str, str]]:
    messages = [turn_to_chat_message(turn) for turn in pair.context_turns]
    messages.extend(
        [
            {"role": "user", "content": pair.input},
            {"role": "assistant", "content": pair.output},
        ]
    )
    return messages


def write_jsonl(pairs: list[TrainingPair], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for pair in pairs:
            record = {
                "messages": build_jsonl_messages(pair),
                "metadata": {
                    "chat_identifier": pair.chat_identifier,
                    "prompt_message_count": pair.prompt_message_count,
                    "response_message_count": pair.response_message_count,
                    "context_turn_count": len(pair.context_turns),
                    "context_message_count": sum(
                        turn.message_count for turn in pair.context_turns
                    ),
                },
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract iMessage conversations into supervised fine-tuning pairs."
    )
    parser.add_argument(
        "--chat-ids",
        "--chat_ids",
        dest="chat_ids",
        help='Comma-separated chat identifiers, for example "+14405540448,+15551234567".',
    )
    parser.add_argument(
        "--chat-ids-file",
        type=Path,
        help="File containing one chat identifier per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path.home() / "Library" / "Messages" / "chat.db",
        help="Path to the Messages SQLite database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Most recent messages to read per chat. Use -1 for all matching messages.",
    )
    parser.add_argument(
        "--min-date",
        help="Only include messages on or after this date or datetime, for example 2025-01-01.",
    )
    parser.add_argument(
        "--max-date",
        help="Only include messages on or before this date or datetime, for example 2025-12-31.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="Minimum character length for both input and output turns.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        help="Maximum character length for both input and output turns.",
    )
    parser.add_argument(
        "--min-output-chars",
        type=int,
        help=(
            "Minimum character length for output turns only. Defaults to "
            "--min-chars when omitted."
        ),
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        help=(
            "Maximum character length for output turns only. Defaults to "
            "--max-chars when omitted."
        ),
    )
    parser.add_argument(
        "--max-pairs-per-chat",
        type=int,
        help="Maximum training pairs to export from each chat after filtering.",
    )
    parser.add_argument(
        "--context-turns",
        type=int,
        default=0,
        help=(
            "Number of previous conversation turns to prepend to each JSONL "
            "record. CSV output remains prompt/response only."
        ),
    )
    parser.add_argument(
        "--strip-urls",
        action="store_true",
        help="Remove URLs during export. By default text style is preserved.",
    )
    parser.add_argument(
        "--include-reactions",
        action="store_true",
        help="Include iMessage tapbacks/reactions. Reactions are skipped by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_pairs.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        help="Optional JSONL output path using chat-style messages for SFT.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.limit < -1:
        parser.error("--limit must be -1 or greater.")
    if args.min_chars < 0:
        parser.error("--min-chars must be 0 or greater.")
    if args.max_chars is not None and args.max_chars < args.min_chars:
        parser.error("--max-chars must be greater than or equal to --min-chars.")
    if args.min_output_chars is not None and args.min_output_chars < 0:
        parser.error("--min-output-chars must be 0 or greater.")
    effective_min_output_chars = (
        args.min_chars if args.min_output_chars is None else args.min_output_chars
    )
    if (
        args.max_output_chars is not None
        and args.max_output_chars < effective_min_output_chars
    ):
        parser.error(
            "--max-output-chars must be greater than or equal to the effective "
            "minimum output length."
        )
    if args.max_pairs_per_chat is not None and args.max_pairs_per_chat < 1:
        parser.error("--max-pairs-per-chat must be 1 or greater.")
    if args.context_turns < 0:
        parser.error("--context-turns must be 0 or greater.")

    chat_identifiers = read_chat_ids(args.chat_ids, args.chat_ids_file)
    if not chat_identifiers:
        parser.error("Provide at least one chat identifier with --chat-ids or --chat-ids-file.")

    min_date = parse_apple_timestamp(args.min_date) if args.min_date else None
    max_date = (
        parse_apple_timestamp(args.max_date, end_of_day=True) if args.max_date else None
    )

    if not args.db_path.exists():
        parser.error(f"Messages database not found: {args.db_path}")

    all_pairs: list[TrainingPair] = []
    totals = {
        "raw_rows": 0,
        "candidate_rows": 0,
        "rows_fetched": 0,
        "messages": 0,
        "turns": 0,
        "pairs": 0,
        "pairs_dropped_by_cap": 0,
        "decoded_attributed_rows": 0,
        "excluded_reaction_rows": 0,
        "skipped_decoded_reaction_rows": 0,
        "skipped_undecoded_rows": 0,
        "skipped_empty": 0,
        "skipped_by_length": 0,
    }

    db_uri = f"file:{quote(str(args.db_path.resolve()), safe='/')}?mode=ro"
    try:
        connection = sqlite3.connect(db_uri, uri=True)
    except sqlite3.OperationalError as error:
        parser.error(f"Unable to open Messages database: {error}")

    try:
        cursor = connection.cursor()
        for chat_identifier in chat_identifiers:
            result = fetch_messages(
                cursor,
                chat_identifier,
                limit=args.limit,
                min_date=min_date,
                max_date=max_date,
                include_reactions=args.include_reactions,
            )
            messages = result.messages
            turns, skipped_empty = group_turns(messages, strip_urls=args.strip_urls)
            pairs, skipped_by_length = build_training_pairs(
                turns,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                min_output_chars=args.min_output_chars,
                max_output_chars=args.max_output_chars,
                context_turns=args.context_turns,
            )
            pairs, pairs_dropped_by_cap = cap_training_pairs(
                pairs,
                args.max_pairs_per_chat,
            )

            all_pairs.extend(pairs)
            totals["raw_rows"] += result.raw_rows
            totals["candidate_rows"] += result.candidate_rows
            totals["rows_fetched"] += result.rows_fetched
            totals["messages"] += len(messages)
            totals["turns"] += len(turns)
            totals["pairs"] += len(pairs)
            totals["pairs_dropped_by_cap"] += pairs_dropped_by_cap
            totals["decoded_attributed_rows"] += result.decoded_attributed_rows
            totals["excluded_reaction_rows"] += result.excluded_reaction_rows
            totals["skipped_decoded_reaction_rows"] += (
                result.skipped_decoded_reaction_rows
            )
            totals["skipped_undecoded_rows"] += result.skipped_undecoded_rows
            totals["skipped_empty"] += skipped_empty
            totals["skipped_by_length"] += skipped_by_length

            print(
                f"{chat_identifier}: "
                f"{result.raw_rows} raw rows, "
                f"{result.excluded_reaction_rows} reactions excluded before limit, "
                f"{result.candidate_rows} text candidates, "
                f"{result.rows_fetched} fetched, {len(messages)} decoded messages, "
                f"{result.skipped_decoded_reaction_rows} decoded reactions skipped, "
                f"{len(turns)} turns, {len(pairs)} pairs"
                f"{f', {pairs_dropped_by_cap} pairs dropped by cap' if pairs_dropped_by_cap else ''}"
            )
    finally:
        connection.close()

    write_csv(all_pairs, args.output)
    if args.jsonl_output:
        write_jsonl(all_pairs, args.jsonl_output)

    print("\nSummary")
    print(f"Chats processed: {len(chat_identifiers)}")
    print(f"Raw rows matched: {totals['raw_rows']}")
    print(f"Rows with text or attributedBody: {totals['candidate_rows']}")
    print(f"Rows fetched after limit: {totals['rows_fetched']}")
    print(f"Decoded messages read: {totals['messages']}")
    print(f"Decoded attributedBody rows: {totals['decoded_attributed_rows']}")
    print(f"Reaction rows excluded before limit: {totals['excluded_reaction_rows']}")
    print(
        "Decoded reaction rows skipped after fetch: "
        f"{totals['skipped_decoded_reaction_rows']}"
    )
    print(f"Skipped undecoded attributedBody rows: {totals['skipped_undecoded_rows']}")
    print(f"Turns built: {totals['turns']}")
    print(f"Training pairs written: {totals['pairs']}")
    print(f"Pairs dropped by per-chat cap: {totals['pairs_dropped_by_cap']}")
    print(f"Skipped empty turns/messages: {totals['skipped_empty']}")
    print(f"Skipped by length filters: {totals['skipped_by_length']}")
    print(f"CSV output: {args.output}")
    if args.jsonl_output:
        print(f"JSONL output: {args.jsonl_output}")


if __name__ == "__main__":
    main()
