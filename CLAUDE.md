# Working in this repo

nbdev. The notebooks under `nbs/` are the source; `anya/*.py` is generated. Edit the notebook, run
`nbdev_export`, never edit the `.py`. CI runs `nbdev_export` and fails on a diff.

`.cursor/install.sh` builds `.venv` with every runtime. Activate it or prefix with `uv run`; nbdev
3.3 names its commands with hyphens, so it is `nbdev-export` and `nbdev-test` there.

## Prose in notebooks

Keep it short. The prose is there so a reader can see what the code does and what was chosen, not to
argue for it.

- Lead with the action or the decision. First sentence does real work. No throat clearing.
- Say what is chosen and move on. One line of reason where the choice is surprising, none where it
  is not.
- Numbers instead of adjectives. "0.406 for a 200-value channel, under the 0.5 floor" beats "the
  scores were low".
- No em dashes, no bold inside a paragraph, no "not just X but Y", no rhetorical questions.
- A design rationale that runs past three sentences belongs in a docstring or in `evals/`, not in a
  markdown cell.

## Docstrings and comments

The code is the document. Prose explains what the code cannot say about itself, and nothing else.

- Docstrings: one line. Add a second sentence only to state a measured number or a footgun.
- Inline comments in a `def` signature are nbdev docments and become the API parameter table. Keep
  them, keep them short.
- Body comments: one line, and only where the code genuinely does not say it.
- No changelog in a comment. "This used to do X and it broke" is a commit message.

## Runtimes

Every runtime is an optional extra. `anya/__init__.py` imports none of them, `core.get_runtime`
imports one lazily, and an ImportError must name the extra that fixes it (`pip install 'anya[onnx]'`).

A runtime subclass supplies three things and inherits the rest: `_read_spec` (what the graph
declares), `_infer` (run one batch), and `_mk_prep` where the signature alone is not enough.
`Model._finish` resolves the labels, the task and the `Prep` for all of them; `_own_labels` and
`_guess_task` are the other two hooks, and only Core ML and LiteRT need any of them. Preprocessing,
batching, decoding, error handling and file arrangement live in `core` and `vision`, once.

Nothing in `anya/vision.py` may import a runtime. It is numpy and Pillow so that the decoders can be
tested without a wheel.

## Tests and fixtures

Tests are non-exported `#| hide` cells. They run against the tiny models in `nbs/fixtures`, which are
built by `evals/mkfixtures.py` and committed, so CI needs the inference runtimes but no download and
no tensorflow. Regenerate them only when a fixture needs to change shape, and say so in the commit.

Anything that needs the network or a Mac is `#| eval: false`. `04_apple` is entirely so.

## Two traps worth knowing

`sort_images` writes into a folder that is usually inside the folder it just read. `items(exclude=)`
is what stops the second run from reading the first run's output. Keep it wired.

A model's signature says the shape and the dtype. It does not say the normalisation, and guessing
wrong gives confident nonsense rather than an error. `norm=` names it, `anya.hub.prep_kwargs` reads
it from the repo's `preprocessor_config.json`, and LiteRT infers it from the input quantisation.
