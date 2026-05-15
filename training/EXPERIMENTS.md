# Seanbot Experiments

This log tracks durable experiment conclusions. Keep raw eval outputs and private
generated text under `eval_runs/`, which is ignored by Git. Summarize only
non-sensitive patterns here.

## Eval Rubric

Use 1-5 scores, where 5 is best.

- `tone`: sounds like Sean's texting style without feeling forced.
- `coherence`: answers the actual prompt and stays internally consistent.
- `specificity`: gives concrete content instead of generic evasions.
- `length_fit`: response length feels plausible for the prompt.
- `not_weird`: avoids random, overconfident, or contextless claims.
- `stop_clean`: stops after one response with no template/transcript leakage.

## Current Baseline

Training setup for A/B/C comparison:

- Model: `unsloth/Meta-Llama-3.1-8B-bnb-4bit`
- Steps: `120`
- Decode: `temperature 0.65`, `top_p 0.9`
- Eval prompts: `training/eval_prompts.txt`
- Loss: completion-only, final assistant response target only

## Results

| Run | Dataset Recipe | Tone | Coherence | Specificity | Length Fit | Not Weird | Stop Clean | Conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | balanced 1:1, `--min-chars 2` | 3 | 3 | 2 | 2 | 3 | 5 | Useful baseline; too terse and sometimes randomly specific. |
| B | A + `--min-output-chars 12` | 3 | 3 | 3 | 4 | 3 | 5 | Better than A; less terse and more useful. |
| C | B + `--context-turns 2` | 4 | 3 | 3 | 4 | 3 | 5 | Best overall so far; more conversational, but still invents context. |

## Experiment Notes

### A: `balanced-1to1`

Command shape:

```bash
uv run main.py \
  --limit 5000 \
  --max-pairs-per-chat 300 \
  --min-date 2025-01-01 \
  --min-chars 2 \
  --max-chars 2000 \
  --jsonl-output exp-a-balanced-1to1.jsonl
```

Observed behavior:

- Clean stopping; no prompt-template leakage.
- Texting style is recognizable enough for a baseline.
- Median response length was very short.
- Many outputs were low-information replies.
- Some prompts produced oddly specific invented context.

Conclusion:

- Dataset balance helped, but short response targets made the model too terse.

### B: `balanced-min-response`

Command shape:

```bash
uv run main.py \
  --limit 5000 \
  --max-pairs-per-chat 300 \
  --min-date 2025-01-01 \
  --min-chars 2 \
  --min-output-chars 12 \
  --max-chars 2000 \
  --jsonl-output exp-b-balanced-min-response.jsonl
```

Observed behavior:

- Clearly less terse than A.
- Better plans/status/logistics replies.
- More useful response length without obvious tone regression.
- Still invents context on memory/open-ended prompts.
- Still has occasional mismatched social/emotional tone.

Conclusion:

- `--min-output-chars 12` should be treated as the current baseline filter.

### C: `balanced-context`

Command shape:

```bash
uv run main.py \
  --limit 5000 \
  --max-pairs-per-chat 300 \
  --min-date 2025-01-01 \
  --min-chars 2 \
  --min-output-chars 12 \
  --max-chars 2000 \
  --context-turns 2 \
  --jsonl-output exp-c-balanced-context.jsonl
```

Hypothesis:

- Including prior turns should reduce random-but-plausible replies by giving
  the model more local conversational grounding.

Risks:

- Context may encourage transcript continuation if rendering/masking is wrong.
- Longer examples may require more training steps or sequence length.

Observed behavior:

- Most conversational and natural of A/B/C.
- Longer than B without obvious stopping/template regressions.
- Better simple greetings, plans/status, logistics, and support prompts.
- Still invents context on memory, advice, and ambiguous prompts.
- Context improves conversational feel more than factual grounding.

Conclusion:

- `--context-turns 2` should become the current baseline recipe.
- The next issue to attack is unsupported invention, not response length.

## Next Ideas

- Keep `--min-output-chars 12` and `--context-turns 2` as the current baseline recipe.
- Add more 1:1 chats before increasing model size.
- Add eval prompts that specifically test unsupported invention and uncertainty.
- Consider prompt wording that encourages uncertainty when context is missing.
- Once the dataset recipe stabilizes, try the same recipe on a larger model.
