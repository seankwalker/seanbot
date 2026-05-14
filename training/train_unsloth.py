import argparse
import os
from pathlib import Path


DEFAULT_MODEL = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
DEFAULT_DATASET = "seankwalker/seanbot-2-imessage"
DEFAULT_HUB_MODEL_ID = "seankwalker/seanbot-2-llama-3-1-imessage"
DEFAULT_SYSTEM_MESSAGE = (
    "You are a 28 year old male named Sean, having a conversation with a friend"
)

PROMPT_TEMPLATE = """Below are some statements that have been made by the other person in a conversation with you. Write responses that appropriately respond to each message.

### Statement:
{INPUT}

### Response:
{OUTPUT}"""


def render_prompt(input_text: str, output_text: str, system_message: str) -> str:
    prompt = PROMPT_TEMPLATE.replace("{INPUT}", input_text).replace(
        "{OUTPUT}",
        output_text,
    )
    if system_message:
        return f"{system_message}\n\n{prompt}"
    return prompt


def render_messages(messages: list[dict[str, str]], system_message: str) -> str:
    user_messages = [
        message["content"]
        for message in messages
        if message.get("role") in {"user", "human"}
    ]
    assistant_messages = [
        message["content"]
        for message in messages
        if message.get("role") in {"assistant", "gpt"}
    ]

    if not user_messages or not assistant_messages:
        raise ValueError("messages rows must include user and assistant content.")

    return render_prompt(
        "\n\n".join(user_messages),
        "\n\n".join(assistant_messages),
        system_message,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Llama 3.1 model on iMessage prompt/response pairs."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)

    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name with input/output columns.",
    )
    data_source.add_argument(
        "--dataset-file",
        type=Path,
        help="Local CSV or JSONL dataset exported by imessage/main.py.",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-num-proc", type=int, default=2)
    parser.add_argument("--conversation-extension", type=int, default=3)
    parser.add_argument("--system-message", default=DEFAULT_SYSTEM_MESSAGE)

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)

    parser.add_argument("--sample-prompt", help="Optional prompt to test after training.")
    parser.add_argument("--sample-max-new-tokens", type=int, default=80)

    parser.add_argument(
        "--save-gguf-dir",
        type=Path,
        help="Optional local directory for a GGUF export.",
    )
    parser.add_argument(
        "--push-gguf",
        action="store_true",
        help="Push a GGUF export to Hugging Face Hub after training.",
    )
    parser.add_argument("--hub-model-id", default=DEFAULT_HUB_MODEL_ID)
    parser.add_argument("--gguf-quantization", default="q8_0")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    return parser


def require_cuda_runtime() -> None:
    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "PyTorch is not installed. Install the Colab/Unsloth dependencies first."
        ) from error

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")
        return

    raise SystemExit(
        "No CUDA GPU detected. In Colab, use Runtime > Change runtime type > GPU, "
        "then restart the runtime, reinstall dependencies, and rerun this script. "
        f"Current PyTorch: {torch.__version__}"
    )


def load_source_dataset(args: argparse.Namespace):
    from datasets import load_dataset

    if args.dataset_file is None:
        return load_dataset(args.dataset, split=args.dataset_split)

    suffix = args.dataset_file.suffix.lower()
    if suffix == ".csv":
        return load_dataset("csv", data_files=str(args.dataset_file), split="train")
    if suffix in {".jsonl", ".json"}:
        return load_dataset("json", data_files=str(args.dataset_file), split="train")

    raise SystemExit(f"Unsupported dataset file type: {args.dataset_file}")


def prepare_dataset(dataset, tokenizer, args: argparse.Namespace):
    from unsloth import apply_chat_template, standardize_sharegpt, to_sharegpt

    columns = set(dataset.column_names)

    if "messages" in columns:
        return dataset.map(
            lambda row: {"text": render_messages(row["messages"], args.system_message)},
            remove_columns=dataset.column_names,
            num_proc=args.dataset_num_proc,
        )

    if {"input", "output"}.issubset(columns):
        dataset = to_sharegpt(
            dataset=dataset,
            merged_prompt="The input is: {input}",
            output_column_name="output",
            conversation_extension=args.conversation_extension,
        )
        dataset = standardize_sharegpt(dataset)
        return apply_chat_template(
            dataset=dataset,
            tokenizer=tokenizer,
            chat_template=PROMPT_TEMPLATE,
            default_system_message=args.system_message,
        )

    raise SystemExit(
        "Dataset must contain either a messages column or input/output columns."
    )


def build_model(args: argparse.Namespace):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def build_training_args(args: argparse.Namespace):
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    training_kwargs = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": args.logging_steps,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "output_dir": args.output_dir,
        "report_to": "none",
    }

    if args.num_train_epochs is None:
        training_kwargs["max_steps"] = args.max_steps
    else:
        training_kwargs["max_steps"] = -1
        training_kwargs["num_train_epochs"] = args.num_train_epochs

    return TrainingArguments(**training_kwargs)


def run_sample(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    system_message: str,
    text_streamer_cls=None,
    fast_language_model_cls=None,
) -> None:
    if text_streamer_cls is None:
        from transformers import TextStreamer

        text_streamer_cls = TextStreamer
    if fast_language_model_cls is None:
        from unsloth import FastLanguageModel

        fast_language_model_cls = FastLanguageModel

    fast_language_model_cls.for_inference(model)
    prompt_text = render_prompt(prompt, "", system_message)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda")

    text_streamer = text_streamer_cls(tokenizer, skip_prompt=True)
    model.generate(
        input_ids,
        streamer=text_streamer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )


def export_gguf(model, tokenizer, args: argparse.Namespace) -> None:
    if args.save_gguf_dir:
        model.save_pretrained_gguf(
            str(args.save_gguf_dir),
            tokenizer,
            quantization_method=args.gguf_quantization,
        )

    if args.push_gguf:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN in the environment before using --push-gguf.")

        model.push_to_hub_gguf(
            args.hub_model_id,
            tokenizer,
            quantization_method=args.gguf_quantization,
            token=token,
            private=args.private,
        )


def main() -> None:
    args = build_parser().parse_args()

    require_cuda_runtime()
    import unsloth  # noqa: F401
    from trl import SFTTrainer

    model, tokenizer = build_model(args)
    dataset = load_source_dataset(args)
    print(f"Loaded dataset columns: {dataset.column_names}")

    dataset = prepare_dataset(dataset, tokenizer, args)
    print(f"Prepared dataset columns: {dataset.column_names}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=False,
        args=build_training_args(args),
    )

    trainer.train()

    if args.sample_prompt:
        run_sample(
            model,
            tokenizer,
            args.sample_prompt,
            args.sample_max_new_tokens,
            args.system_message,
        )

    export_gguf(model, tokenizer, args)


if __name__ == "__main__":
    main()
