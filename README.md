# anya

> Machine learning models as tools. Load a classifier, a detector or a
> segmenter and run it over a folder.

anya is the vision half of the same idea as
[rishi](https://github.com/vedicreader/rishi): one callable over several
runtimes, small enough that a local model can drive it as a tool.
`Model(name)` picks the runtime from the shape of the name, reads the
preprocessing off the graph, and returns something JSON-shaped.

## Install

``` sh
pip install 'anya[onnx]'      # ONNX Runtime, everywhere
pip install 'anya[litert]'    # .tflite, the on-device zoo
pip install 'anya[coreml]'    # .mlpackage on a Mac
pip install 'anya[pii]'       # scikit-learn, to fit a PII detector on your own examples
pip install 'anya[all]'       # every runtime your platform supports, plus hub, video, audio and pii
```

Runtime modules import lazily, so `import anya` never pulls in a wheel you
did not install.

**Contributors:** `pip install -e '.[dev]'` then `nbdev_prepare`. Notebooks
in `nbs/` are the source; `anya/*.py` is generated.

## Three lines

The examples here run against a toy model in `nbs/fixtures` that averages
each colour channel and calls it a class, so the docs execute offline. Swap
the path for a hub repo id and nothing else changes.

``` python
import numpy as np
from anya import Model, classify, sort_images

m = Model('fixtures/tiny_cls_labels.tflite')     # or 'litert-community/some-classifier'
m
```

    LitertModel(tiny_cls_labels.tflite, runtime=litert, task=classify, 4 labels)

``` python
green = np.zeros((64, 64, 3), np.uint8); green[..., 1] = 255
m(green)
```

`array` → **green** (1.000)

- red 0.000
- blue 0.000
- none 0.000

Nothing was passed but the path. The class names came out of the metadata in
the `.tflite` file, the input size and the channel order came out of the
graph, and the task came from the shape of the output.

## Pick a runtime

`Model(name)` routes on the name; `model.runtime` says which one you got.
Force it with `runtime=` or a prefix.

| name looks like                  | runtime                          |
|----------------------------------|----------------------------------|
| `.tflite`, `litert-community/…`  | litert                           |
| `.onnx`, `onnx-community/…`      | onnx                             |
| `.mlpackage`, `.mlmodel`         | coreml                           |
| `org/repo` with nothing to go on | the hub is asked what it ships   |

``` python
from anya.core import resolve_runtime
resolve_runtime('models/yolo11n.onnx'), resolve_runtime('litert-community/birds'), resolve_runtime('org/repo')
```

    (('onnx', 'models/yolo11n.onnx'),
     ('litert', 'litert-community/birds'),
     (None, 'org/repo'))

## A folder in, a sorted folder out

The job this exists for. `classify` over a folder gives `Preds`, which knows
how to tally itself; `sort_images` turns that into files on disk, and plans
before it moves anything.

``` python
ps = classify(d, m, topk=1)
ps.counts(), len(ps.failed)
```

    ({'blue': 2, 'green': 2, 'red': 2}, 1)

``` python
plan = sort_images(d, m)                     # dry run: says what it would do
plan.labels, plan.n, plan.moved
```

    (['blue', 'green', 'red'], 7, 0)

``` python
sort_images(d, m, dry_run=False).moved       # and now it does it
```

    7

One unreadable file does not end a run of 2000. It comes back as a `Pred`
with an `error`, and `arrange` files it under `unsorted/`.

``` python
ps.failed[0]['error']
```

    "UnidentifiedImageError: cannot identify image file '/tmp/tmp2yed_bgd/broken.png'"

## What comes back

A `Pred` is a dict, so it crosses a tool call without a serialiser, and it
answers `.label` and `.score` whatever the task was.

``` python
p = m(green)
p['task'], p.label, p.score, p.preds[:2]
```

    ('classify',
     'green',
     1.0,
     [{'label': 'green', 'score': 1.0, 'index': 1},
      {'label': 'red', 'score': 0.0, 'index': 0}])

## Tools for rishi

`anya.tools` is the same jobs shaped for a model to call: strings in, capped
JSON out, and anything that writes takes `apply=False` by default.

``` python
from rishi import Chat
from anya.tools import TOOLS

chat = Chat(tools=TOOLS)
chat('Sort ~/Pictures/birds into folders by species, and tell me what you found.')
```

``` python
from anya.tools import tool_names
tool_names()
```

    ['find_model',
     'model_info',
     'count_images',
     'classify_image',
     'detect_image',
     'segment_image',
     'classify_folder',
     'similar_images',
     'label_video',
     'scan_text',
     'redact_text',
     'sort_folder',
     'name_model',
     'fit_pii']

There is a CLI over the same functions:

``` sh
anya model_info litert-community/some-classifier
anya classify_folder ~/Pictures/birds --model=aussie-birds
anya sort_folder ~/Pictures/birds --model=aussie-birds --apply=1
```

## PII, fitted on your own examples

An organisation’s PII is mostly not the world’s PII. `EMP-483920` identifies
somebody and no pattern bank ships it; `ORD-483920` identifies a pallet and
has the same shape. `anya.pii` fits a span tagger and a document classifier on
whatever a team already labelled, and defers to arithmetic for the kinds
arithmetic already knows.

``` python
from anya.pii import dataset, evaluate, fit

ds = dataset('fixtures/pii_org.jsonl')       # jsonl, csv, doccano, Label Studio, or a folder of classes
tr, va = ds.split(0.25)                      # holds out whole `group`s, so no template straddles it
det = fit(tr)
det
```

    PiiDetector(mode=defer, kinds=9, deferred=4, doc=yes, bias=0)

``` python
det.report('Escalated by EMP-774310 after the second call.')
```

    {'has_pii': True,
     'kinds': {'emp_id': 1},
     'n': 1,
     'scanned': 46,
     'density': 21.739,
     'label': 'pii',
     'label_score': 0.9129,
     'fitted': True,
     'mode': 'defer',
     'spans': [{'start': 13, 'end': 23, 'kind': 'emp_id',
                'text': 'EMP-774310', 'source': 'learned'}]}

`source` says which layer found it. The patterns cannot see `EMP-774310`; the
tagger cannot see a checksum. On `evals/mkpii.py` the two together score 0.907
span F1 against the pattern bank’s 0.399, and invent nothing the bank had gated
(`evals/RESULTS.md`).

With nothing to fit on you get the same object, answering out of the pattern
bank. That is the fallback: a state, not a second code path.

``` python
fit([]), fit([]).spans('mail jane@example.com')
```

    (PiiDetector(unfitted, baseline=floor_spans),
     [(5, 21, 'email', 'jane@example.com')])

`fit_run` is the loop: split, fit, sweep the decoding bias, score, compare
against the baseline with a paired bootstrap, and write a run to
`~/.anya/pii/runs/` with a `report.html` you can open. It draws per-kind
precision and recall, what a point of recall costs, and every mistake with the
characters either side of it, filterable. In a notebook the returned run
renders as that report.

``` python
from anya.pii import fit_run
run = fit_run('tickets.jsonl', name='acme', save_as='acme')   # then: anya scan_text "..." --model=acme
run                                                           # the report, inline
```

## Finding a model

anya ships no default model and no alias table: a classifier for Australian
birds and one for chest X-rays are both "classify", and a repo id that does
not exist is worse than a search.

``` python
from anya.hub import find_models, alias
find_models('bird classifier', task='classify', runtime='litert')
alias('aussie-birds', 'org/whatever-you-picked', file='model.tflite')
Model('aussie-birds')          # from now on, by name
```

`find_models` asks the Hub; with no Hub reachable, `web_models` asks the open
web through [fossick](https://github.com/vedicreader/fossick) instead.

## Modules

| notebook     | for                                                                              |
|--------------|----------------------------------------------------------------------------------|
| `00_core`    | `items`, `Pred`/`Preds`, runtime resolution, the `Model` base, `arrange`          |
| `01_vision`  | loading, preprocessing, and the decoders (softmax, NMS, masks)                    |
| `02_onnx`    | ONNX Runtime                                                                     |
| `03_litert`  | `.tflite` over LiteRT, quantisation and metadata labels                          |
| `04_apple`   | Core ML on macOS                                                                 |
| `05_hub`     | finding, fetching and naming models                                              |
| `06_tasks`   | `classify` / `detect` / `segment` / `embed`, `sort_images`, `find_similar`, `bench` |
| `07_tools`   | the tool surface a chat model calls, and the CLI                                 |
| `08_pii`     | fitting a PII detector on your own examples, and the fallback when you have none |
| `09_runs`    | where a training run’s numbers live, and the report you read them on             |
