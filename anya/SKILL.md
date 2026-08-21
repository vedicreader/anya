---
name: anya
description: Run vision models as tools - load a .tflite/.onnx/.mlpackage classifier, detector, segmenter or embedder with anya's Model API, run it over one picture or a folder of thousands, and sort the folder by what came back. Use when a task involves classifying, detecting, segmenting, embedding or tidying images, video frames or audio clips with a local model, or anything mentioning anya, litert/tflite inference, onnxruntime vision models, or Core ML .mlpackage models.
---

# anya

anya loads a vision model and runs it over things. `Model(name)` picks the runtime from the shape of
the name and returns that runtime's subclass, the way `rishi.Chat` picks a chat backend. There is no
default model: anya runs the model you name.

```python
from anya import Model, classify, sort_images
m = Model('litert-community/some-classifier')   # or a path, or an alias
m('bird.jpg')                                   # -> Pred
classify('~/Pictures/birds', m)                 # -> Preds, one per file
```

## The one thing to remember

`m(x)` returns a `Pred` for one item and `Preds` for a folder. A `Pred` is a dict (so it JSON-dumps
as is) that also answers `.label` and `.score` whatever the task was. The task-shaped payload is
under `preds` (classify), `objects` (detect), `mask`/`classes` (segment) or `vec` (embed).

## API surface

- `Model(model=None, *, runtime=None, model_path=None, file=None, revision=None, task=None, labels=None, norm=None, size=None, resize=None, prep=None, topk=5, conf=0.25, iou=0.45, **runtime_kw)` - `model` first (a path, a hub repo id, or an alias). Dispatches to `OnnxModel` / `LitertModel` / `CoreMLModel`. Runtime-specific keywords pass through: `providers=`/`max_bs=` (onnx), `threads=`/`max_bs=` (litert), `compute_units=` (coreml).
- `m(x)`, `m.predict(one)`, `m.predict_all(folder, bs=None, on_error='skip', exclude=None)`, `m.predict_video(path, every=1.0, max_frames=None)`.
- State: `m.task`, `m.runtime`, `m.labels`, `m.prep`, `m.spec`, `m.max_bs`, `m.modality`. `m.close()` frees the graph.
- `Preds`: `.counts()`, `.above(score)`, `.by_label()`, `.ok`, `.failed`, `.labels`, `.records()`, `.save(path)`.
- Tasks (`anya.tasks`): `classify(x, model, topk=5)`, `detect(x, model, conf, iou)`, `segment(x, model)`, `embed(x, model)`, `run(x, model, task=None)`, `ask(x, {'find': 'car', 'task': 'segment'})`, `pick(pred, want)`, `save_masks(pred, dest, want=None)`, `sort_images(folder, model, dest=None, min_score=0.5, how='copy', dry_run=True)`, `find_similar(query, folder, model, n=10)`, `index_folder`, `summarize(preds)`, `bench(model)`.
- Files (`anya.core`): `items(o, types='image', exclude=None)` expands a folder/glob/list; `arrange(preds, dest, how='copy', min_score=0, dry_run=True)` files them by label; `load_model` caches by arguments; `safe_name` turns a label into a directory name.
- Hub (`anya.hub`): `find_models(query, task=, runtime=)`, `web_models(query)` (fossick fallback), `resolve_model(repo_id)`, `fetch(repo, file)`, `alias(name, repo, **kw)`, `aliases()`.
- Tools (`anya.tools`): `TOOLS` for `Chat(tools=...)`, `SAFE` for the read-only subset, `segment_masks` to write one PNG per class, plus the `anya` CLI over the same functions.

## Runtimes

| name looks like | runtime | extra |
|---|---|---|
| `.tflite`, `litert-community/…` | litert | `pip install 'anya[litert]'` |
| `.onnx`, `onnx-community/…` | onnx | `pip install 'anya[onnx]'` |
| `.mlpackage`, `.mlmodel` | coreml | `pip install 'anya[coreml]'`, macOS |
| `org/repo` with nothing to go on | the hub decides from the files it ships | `pip install 'anya[hub]'` |

`runtime=` or a `litert/…` prefix forces it. `resolve_runtime(name)` gives the decision without
building anything.

## Preprocessing is read, not written

The graph says the input size, the channel axis, the dtype and the quantisation, so `Prep` is built
from the signature. What a graph cannot say is the normalisation:

- `norm='01'` (default) for 0..1 pixels, `'imagenet'` for torchvision and timm exports, `'signed'`
  for -1..1, `'none'` for raw 0..255, or an explicit `(mean, std)`.
- LiteRT infers it from the input quantisation: `scale=1/255, zero_point=0` means 0..1.
- A HuggingFace repo's `preprocessor_config.json` fills in `size`, `norm` and `center_crop` on its
  own, and `config.json`'s `id2label` supplies the labels.

Getting this wrong gives confident wrong answers rather than an error. If a model's top label looks
random, try `norm='imagenet'` or `norm='signed'` before anything else.

## Masks come back in the picture's pixels

`segment` argmaxes the logits, then un-maps them. Letterbox bars come off and the label map is
resized to the picture that went in, as detection boxes already were. A segformer with a
128x128 logit grid asked about a 724x543 photo returns a 543x724 label map.

Getting one out of the process takes `save_masks(pred, dest, want='car')`, or the `segment_masks`
tool, which writes `<stem>.<label>.png` per class and returns the paths. The default destination is a
`masks` folder beside the picture. A later folder run over that folder needs `exclude=`, or a
destination somewhere else.

`ask` is the same jobs behind one small dict, for an agent that is filling in a form rather than
writing Python:

```python
from anya import Model
from anya.tasks import ask
ask('frontage.png', {'find': 'car', 'task': 'segment', 'model': 'Xenova/segformer-b0-finetuned-cityscapes-1024-1024'})
# -> Pred with `classes` cut down to the car, `hit` a boolean mask for it, `mask` still the full label map
```

A coarse mask is normal. A semantic segmenter at 512x512 gives a blocky edge, and refining it onto
the real one is the editing tool's job rather than anya's.

## Labels

In order: `labels=` you pass (a list, a `labels.txt`, or a `config.json`), then the repo's
`config.json`, then a `.tflite` file's own metadata zip, then a labels file beside the weights. With
none of those the classes come back as `class_0`, `class_1`.

## Sorting a folder

```python
sort_images('~/Pictures/birds', 'aussie-birds')                  # plan only
sort_images('~/Pictures/birds', 'aussie-birds', dry_run=False)   # do it
```

Default destination is `<folder>/sorted/<label>/`. Low-confidence and failed items go to
`unsorted/`. `how='copy'` (default), `'move'`, or `'link'` for a hard link. The scan skips the
destination, so a second run does not re-read the first run's output.

## Gotchas

- A folder read is recursive and image-only by default. `types='any'` or `types='audio'` changes it.
- One unreadable file becomes a `Pred` with `error` and does not end the run. `on_error='raise'`
  if you would rather it did.
- Most `.tflite` graphs are batch-1. `max_bs=8` asks the interpreter to resize its input tensor and
  falls back to 1 when it will not.
- Detection boxes come back as `[x1, y1, x2, y2]` in the original image's pixels, already
  un-letterboxed. Scores are after NMS.
- `decode_yolo` reads the layout from the label count when labels are known. Without labels it
  guesses whether column 0 is objectness, so pass `labels=` for a v5-era export.
- Core ML classifiers return `{label: probability}` themselves, so `labels=` is ignored there.

## Working on anya itself

nbdev. Edit `nbs/*.ipynb`, not `anya/*.py`, then run `nbdev_prepare`. Tests are `#| hide` cells over
the tiny committed models in `nbs/fixtures`; `04_apple` is `#| eval: false` because CI is linux.

Docs: https://vedicreader.github.io/anya/
