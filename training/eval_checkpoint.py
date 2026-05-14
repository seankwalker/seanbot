import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_unsloth import (  # noqa: E402
    DEFAULT_SYSTEM_MESSAGE,
    DISABLE_RESPONSE_END_MARKER,
    generate_response,
    require_cuda_runtime,
    resolve_response_end_marker,
)


DEFAULT_PROMPTS_FILE = Path(__file__).with_name("eval_prompts.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeatable prompt evals against a trained Seanbot checkpoint."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Local checkpoint directory or Hugging Face model/adapter name.",
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-label", help="Optional label for this eval run.")

    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stochastic sampling. Use --no-do-sample for greedy decoding.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--system-message", default=DEFAULT_SYSTEM_MESSAGE)
    parser.add_argument(
        "--response-end-marker",
        default=None,
        help=(
            "Text that marks response end. Defaults to tokenizer.eos_token. "
            'Pass "" to disable.'
        ),
    )
    return parser


def read_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        prompt = line.strip()
        if not prompt or prompt.startswith("#"):
            continue
        prompts.append(prompt)
    return prompts


def prepare_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output already exists. Use --overwrite to replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        path.write_text("", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(args: argparse.Namespace):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)
    return model, tokenizer, FastLanguageModel


def build_metadata(
    args: argparse.Namespace,
    prompts: list[str],
    created_at: str,
) -> dict:
    return {
        "run_label": args.run_label,
        "created_at": created_at,
        "model_name": args.model_name,
        "prompts": str(args.prompts),
        "prompt_count": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "generation": {
            "max_new_tokens": args.sample_max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "seed": args.seed,
        },
        "system_message": args.system_message,
        "response_end_marker": args.response_end_marker,
    }


def build_record(
    metadata: dict,
    prompt_index: int,
    sample_index: int,
    prompt: str,
    response: str,
) -> dict:
    return {
        "prompt_index": prompt_index,
        "sample_index": sample_index,
        "prompt": prompt,
        "response": response,
        "metadata": metadata,
    }


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_eval(args: argparse.Namespace) -> None:
    require_cuda_runtime()
    set_seed(args.seed)

    prompts = read_prompts(args.prompts)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")

    prepare_output_path(args.output, args.overwrite)
    model, tokenizer, fast_language_model_cls = load_model(args)
    response_end_marker = resolve_response_end_marker(
        tokenizer,
        args.response_end_marker,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    metadata = build_metadata(args, prompts, created_at)
    metadata["response_end_marker"] = (
        response_end_marker
        if response_end_marker != DISABLE_RESPONSE_END_MARKER
        else None
    )

    for prompt_index, prompt in enumerate(prompts):
        for sample_index in range(args.samples_per_prompt):
            response = generate_response(
                model,
                tokenizer,
                prompt,
                args.sample_max_new_tokens,
                args.system_message,
                response_end_marker=response_end_marker,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                fast_language_model_cls=fast_language_model_cls,
            )
            record = build_record(
                metadata,
                prompt_index,
                sample_index,
                prompt,
                response,
            )
            append_jsonl(args.output, record)
            print(f"[{prompt_index + 1}/{len(prompts)}] {prompt}")
            print(f"> {response}\n")

    print(f"Wrote eval results to {args.output}")


def main() -> None:
    run_eval(build_parser().parse_args())


if __name__ == "__main__":
    main()
