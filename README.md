# chiroti

Python client for Chiroti, the lab's shared LLM server. Talk to whatever
model is currently hosted on the DGX Spark, from your own laptop, with no
URLs, HTTP, or JSON in your code.

## Install

```bash
pip install git+https://github.com/rythdg/chiroti-client.git
```

## Configure

Set these once (env vars, or a `~/.chiroti/config.json` file, or a
`chiroti.configure()` call — see precedence below):

```bash
export CHIROTI_SERVER="http://chiroti:8100"
export CHIROTI_TOKEN="<ask Ryth for the current token>"
```

or, in code:

```python
import chiroti
chiroti.configure(server="http://chiroti:8100", token="...")
```

Precedence when more than one is set: explicit `configure()` call > env var >
`~/.chiroti/config.json` > (no default — you must set one of these).

## Basic usage

```python
import chiroti

chiroti.ask("Why is the sky blue?")
chiroti.ask("Write a haiku.", temperature=0.9, top_p=0.95, stop=["\n\n"])
```

`chiroti.ask()` forwards standard OpenAI sampling parameters straight through
to the model: `temperature`, `top_p`, `n`, `stop`, `presence_penalty`,
`frequency_penalty`, `seed`, `logprobs`, `top_logprobs`, `max_tokens`.

```python
chiroti.models()   # -> ["qwen-vl-32b"], whichever single model is hosted right now
```

There is exactly one model hosted at a time — the admin swaps it manually.
You don't need to pass `model=` unless you want Chiroti to double-check you're
talking to the model you think you are (it raises a clear error if the name
you pass doesn't match what's currently hosted).

## Giving it data — `data=`

Pass one CSV/NPZ file, or a list of them, and Chiroti turns them into a JSON
block appended to your prompt — the model sees the real structured data, not
a description of it.

```python
chiroti.ask(
    "Summarize the trend and compute the mean voltage.",
    data="trial1.csv",
)

# pass several files at once — each keeps its own data independent of the
# others, keyed by filename; they don't need matching columns
chiroti.ask("Summarize across all trials.", data=["trial1.csv", "trial2.csv", "trial3.csv"])

# .npz arrays work too — small arrays are included in full, large arrays
# are summarized (shape/dtype/min/max/mean/std) so you don't blow the
# model's context window
chiroti.ask("How many spikes, and what's the average inter-spike interval?", data="spikes.npz")
```

Notes:
- Each CSV/NPZ file is kept under its own filename key in the JSON block —
  they're independent of each other and don't need matching columns.
- There's a row cap per CSV file (2000 rows) and an inline-value
  cap for NPZ arrays (200 values) before Chiroti falls back to summary
  statistics instead of the raw values — both fail with a clear error/summary
  rather than silently truncating.
- This is pure client-side text preparation. No file is uploaded anywhere;
  Chiroti just builds a bigger text prompt.

## Getting structured output back — `output_format=`

Pass a Pydantic model class, and `ask()` returns an instance of it instead of
a plain string:

```python
from pydantic import BaseModel

class Summary(BaseModel):
    trend: str
    mean_voltage_mV: float

result = chiroti.ask(
    "Given this voltage trial data, summarize the trend and compute the mean.",
    data="trial1.csv",
    output_format=Summary,
)
result.mean_voltage_mV   # a real float, not a string you have to parse
```

If the currently hosted model doesn't support structured output, you get
`chiroti.exceptions.UnsupportedFeatureError` naming that model. If the model's
output doesn't validate against your schema, you get
`chiroti.exceptions.OutputValidationError` — the raw text it returned is on
`error.raw_text` for debugging.

## What doesn't work yet

`image=`, `document=` (PDF/ZIP), `reasoning=`, and `cache=` are accepted by
`ask()`'s signature but raise `NotImplementedError` if you actually pass a
value — they're reserved names for features not built yet, not silently
ignored.

## Exceptions

Every failure is a specific `chiroti.exceptions.*` class, never a raw
HTTP/connection traceback:

| Exception | Meaning |
|---|---|
| `AuthenticationError` | Your token was missing or wrong. |
| `InvalidInputError` | The request itself was invalid (empty prompt, mismatched CSV columns, unsupported file type, an unknown sampling kwarg, ...). |
| `ModelNotFoundError` | The `model=` you asked for isn't the one currently hosted. |
| `UnsupportedFeatureError` | The currently hosted model can't do what you asked (e.g. `output_format=` on a model without structured-output support). |
| `InferenceError` | The model backend failed, or didn't respond within 300s. |
| `ChirotiConnectionError` | The Chiroti server itself couldn't be reached. |
| `OutputValidationError` | The model's response didn't match your `output_format=` schema (see `.raw_text`). |
