"""Run anya against real models on real photographs, and check the answers.

The notebook tests use the tiny graphs in nbs/fixtures, which prove the plumbing and nothing about
what a real export declares. This downloads eight repos and asserts on what comes back.

    python evals/e2e.py                 # every case
    python evals/e2e.py detect audio    # only the cases whose name contains one of these

Needs the network, and `pip install 'anya[onnx,litert,hub,audio]'`. About 190MB of weights, cached
by huggingface_hub after the first run.
"""
import sys, urllib.request
from pathlib import Path

import numpy as np
from fastcore.all import AttrDict, L

from anya import Model, classify, detect, embed, find_similar, segment, sort_images, summarize
from anya.hub import resolve_model
from anya.tools import classify_image, detect_image, label_video, segment_image, tool
from anya.vision import batch, load_image

CACHE = Path(__file__).parent/'.cache'

# COCO val2017, which every vision library doctests against, plus Ultralytics' two demo pictures
PHOTOS = dict(cats='http://images.cocodataset.org/val2017/000000039769.jpg',      # two cats, two remotes, a sofa
              room='http://images.cocodataset.org/val2017/000000000139.jpg',      # a living room with a television
              bear='http://images.cocodataset.org/val2017/000000000285.jpg',      # one brown bear
              bus='https://ultralytics.com/images/bus.jpg',                       # a bus and three people
              zidane='https://ultralytics.com/images/zidane.jpg')                 # two people, one tie

# ImageNet-1k, COCO-80 and COCO-90 class names, which the repos that need them do not ship
IMAGENET = 'onnx-community/mobilenetv4_conv_small.e2400_r224_in1k'
COCO90 = ('person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant street_sign '
          'stop_sign parking_meter bench bird cat dog sheep horse').split()

# %% ------------------------------------------------------------------ harness

PICS, SORTED = CACHE/'photos', CACHE/'sorted'      # the sorted tree stays outside the folder being read

def photos() -> AttrDict:
    'Download the test photographs once into evals/.cache/photos and return `{name: path}`.'
    PICS.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, u in PHOTOS.items():
        p = PICS/f'{k}{Path(u).suffix}'
        if not p.exists(): p.write_bytes(urllib.request.urlopen(u, timeout=60).read())
        out[k] = p
    return AttrDict(out)

def audio() -> AttrDict:
    'Two seconds each of a 440Hz sine, white noise and silence, which AudioSet has names for.'
    import soundfile as sf
    d = CACHE/'audio'; d.mkdir(parents=True, exist_ok=True)
    t = np.arange(32000)/16000
    for n, x in dict(sine=0.5*np.sin(2*np.pi*440*t), noise=0.2*np.random.default_rng(0).standard_normal(32000),
                     silence=np.zeros(32000)).items():
        if not (d/f'{n}.wav').exists(): sf.write(d/f'{n}.wav', x.astype(np.float32), 16000)
    return AttrDict({n: d/f'{n}.wav' for n in ('sine', 'noise', 'silence')})

CASES, LOG = [], []

def case(f):
    'Register a check. Each one prints what it saw, then asserts.'
    CASES.append(f); return f

def say(*a): LOG.append(' '.join(str(x) for x in a)); print('   ', *a)

def near(a, b, tol=0.06):
    "Two detectors' boxes for the same object, as a fraction of the box's own size."
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return bool((np.abs(a-b) <= tol*max(np.ptp(b[[0, 2]]), np.ptp(b[[1, 3]]))).all())

def biggest(objs, label):
    'The largest box the detector gave `label`, or None.'
    hits = L(objs).filter(lambda o: o['label'] == label)
    return max(hits, key=lambda o: (o['box'][2]-o['box'][0])*(o['box'][3]-o['box'][1]), default=None)

# %% ------------------------------------------------------------------ classify

@case
def classify_onnx(im):
    'A timm export whose every input axis is symbolic and named, at 224 with a 0.875 crop.'
    m = Model(IMAGENET)
    say(m, '|', m.prep)
    assert m.task == 'classify' and m.prep.layout == 'nchw', m.prep      # the names are the only clue
    assert m.prep.size == (224, 224) and m.prep.crop_pct == 0.875, m.prep
    assert m.prep.resample == 3, m.prep                                  # bicubic, as the config asks
    for k, want in dict(cats='tabby', bear='brown bear', bus='bus').items():
        p = m(im[k]); say(f'{k:7s}', p.label, round(p.score, 3))
        assert want in p.label, (k, p.label)
    assert m(im.cats).score > 0.6, m(im.cats).score

@case
def classify_litert(im):
    'A torch export to tflite: NCHW, no config, so the caller supplies the normalisation and names.'
    cfg = Path(resolve_model(IMAGENET)[1]).parent.parent/'config.json'     # 1000 names, from the other repo
    m = Model('litert-community/MobileNet-v3-small', labels=cfg,
              norm='imagenet', resize='center_crop', crop_pct=0.875)
    say(m, '|', m.prep)
    assert m.prep.layout == 'nchw' and m.prep.size == (224, 224), m.prep
    p = m(im.cats); say('cats  ', [(d['label'], round(d['score'], 3)) for d in p.preds[:3]])
    assert 'cat' in p.label, p.label
    assert 'brown bear' in m(im.bear).label, m(im.bear).label

@case
def classify_prep_matches_reference(im):
    'anya\'s tensor against the transform the weights were evaluated with. Needs torch and torchvision.'
    try: import torchvision.transforms as T
    except ImportError: return say('skipped: no torchvision')
    from PIL import Image
    ref = T.Compose([T.Resize(256, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    m = Model(IMAGENET)
    for k in ('cats', 'bus'):
        want = ref(Image.open(im[k]).convert('RGB')).unsqueeze(0).numpy()
        got = np.asarray(batch([load_image(im[k])], m.prep)[0], np.float32)
        say(f'{k:7s} max|diff| {np.abs(want-got).max():.6f}')
        assert np.abs(want-got).max() < 1e-5, np.abs(want-got).max()      # the same tensor, to float32

# %% ------------------------------------------------------------------ detect

@case
def detect_onnx_fp16(im):
    'A float16 yolov8n whose boxes are in input pixels.'
    m = Model('webnn/yolov8n')
    say(m, '|', m.prep, '| declared', m.inp.dtype)
    # the config asks for 224; the graph fixes 640 and will take nothing else
    assert m.prep.size == (640, 640) and m.prep.resize == 'letterbox', m.prep
    assert m.prep.dtype == m.inp.dtype, (m.prep.dtype, m.inp.dtype)
    o = m(im.bus)['objects']; say('bus   ', [(d['label'], round(d['score'], 2)) for d in o[:5]])
    assert L(o).attrgot('label').filter(lambda l: l == 'person').__len__() >= 3, o
    assert biggest(o, 'bus'), o
    c = m(im.cats)['objects']; say('cats  ', [(d['label'], round(d['score'], 2)) for d in c[:5]])
    assert L(c).attrgot('label').filter(lambda l: l == 'cat').__len__() == 2, c
    assert 'remote' in L(c).attrgot('label'), c

@case
def detect_litert_normalised(im):
    'The same (1, 84, 8400) signature as yolov8n, with the boxes 0..1 instead of in pixels.'
    m = Model('SpotLab/YOLOv8Detection')
    ref = Model('webnn/yolov8n')
    say(m, '|', m.prep)
    a, b = biggest(m(im.bus)['objects'], 'bus'), biggest(ref(im.bus)['objects'], 'bus')
    say('bus     litert', [round(v) for v in a['box']], ' onnx', [round(v) for v in b['box']])
    assert near(a['box'], b['box']), (a['box'], b['box'])       # the units were read right, or these disagree
    # this repo's config asks for a 640 centre crop, so the cats are cropped, but both are still there
    labs = L(m(im.cats)['objects']).attrgot('label')
    say('cats   ', list(labs))
    assert len(labs.filter(lambda l: l == 'cat')) == 2, labs

@case
def detect_litert_ssd(im):
    'A quantised efficientdet with a boxes/classes/scores/count head, normalised against the padded input.'
    m = Model('litert-community/efficientdet', labels=COCO90)
    ref = Model('webnn/yolov8n')
    say(m, '|', m.prep)
    assert m.prep.dtype == 'uint8' and m.prep.quant, m.prep
    for k, lab in dict(bus='bus', cats='cat').items():
        a, b = biggest(m(im[k])['objects'], lab), biggest(ref(im[k])['objects'], lab)
        say(f'{k:7s} ssd {[round(v) for v in a["box"]]}  yolo {[round(v) for v in b["box"]]}')
        assert near(a['box'], b['box'], 0.1), (a['box'], b['box'])   # letterbox padding, or y is ~50px out

@case
def detect_raw_head_raises(im):
    'ssdlite320 ships twelve per-stride heads. An error beats the empty list the old rule returned.'
    repo = 'mlboydaisuke/ssdlite320-mobilenetv3-litert'
    try: Model(repo); assert False, 'expected a ValueError at load'
    except ValueError as e: say('at load:', str(e)[:96]); assert 'Pass task=' in str(e), e
    m = Model(repo, task='detect')                                # forced, so the decoder gets to complain
    p = m.predict_all([im.bus])[0]                                # over a list it is recorded, not raised
    say('forced: ', p['error'][:96])
    assert 'Pass task=' in p['error'], p
    d = tool('detect_image')(str(im.bus), repo)
    say('as a tool:', d.get('error', '')[:64])
    assert 'ValueError' in d['error'], d                          # a chat gets JSON, never a traceback

# %% ------------------------------------------------------------------ segment

@case
def segment_onnx(im):
    'SegFormer on ADE20K: 150 classes on a 128x128 grid, put back on the original picture.'
    m = Model('Xenova/segformer-b0-finetuned-ade-512-512')
    say(m, '|', m.prep)
    assert m.task == 'segment' and len(m.labels) == 150, m       # 'logits' does not make it a classifier
    p = m(im.bus)
    say('bus   ', p['shape'], [(c['label'].strip(), c['frac']) for c in p['classes'][:4]])
    assert p['shape'] == list(load_image(im.bus).shape[:2]), p['shape']
    got = L(p['classes']).attrgot('label').map(str.strip)
    assert {'building', 'sidewalk', 'bus'} <= set(got), got
    r = m(im.room); say('room  ', [(c['label'].strip(), c['frac']) for c in r['classes'][:4]])
    assert {'wall', 'floor'} <= set(L(r['classes']).attrgot('label').map(str.strip)), r['classes']

@case
def segment_litert(im):
    'lraspp on VOC: 21 classes, and class 8 is cat.'
    m = Model('litert-community/lraspp_mobilenet_v3_large')
    say(m, '|', m.prep)
    p = m(im.cats)
    by = {c['index']: c['frac'] for c in p['classes']}
    say('cats  ', p['shape'], [(c['index'], c['frac']) for c in p['classes'][:4]])
    assert p['shape'] == list(load_image(im.cats).shape[:2]), p['shape']
    assert by.get(8, 0) > 0.2, by            # VOC 8 is cat, over a fifth of a picture of two cats

# %% ------------------------------------------------------------------ embed

@case
def embed_onnx(im):
    'DINOv2 emits 257 tokens of 384; one vector out means the tokens are pooled, not flattened.'
    m = Model('Xenova/dinov2-small')
    say(m, '|', m.prep)
    assert m.task == 'embed', m               # from the name last_hidden_state, since the shapes cannot say
    p = m(im.cats)
    say('vec', p.vec.shape, 'norm', round(float(np.linalg.norm(p.vec)), 4))
    assert p.vec.shape == (384,), p.vec.shape
    hits = find_similar(im.cats, PICS, m, n=5)
    say('similar', [(Path(h.src).stem, round(h.score, 3)) for h in hits])
    assert Path(hits[0].src).name == im.cats.name, hits[0]          # the query matches itself first
    assert hits[1].score > hits[-1].score, hits

# %% ------------------------------------------------------------------ audio

@case
def audio_litert(im):
    'YAMNet on a sine, noise and silence, with its 521 AudioSet names read out of the file.'
    wav = audio()
    m = Model('thelou1s/yamnet', file='lite-model_yamnet_classification_tflite_1.tflite')
    say(m, '|', m.prep, '| modality', m.modality)
    assert m.modality == 'audio' and m.prep.samples == 15600, m.prep
    assert len(m.labels) == 521, len(m.labels or [])
    got = {}
    for k in ('sine', 'noise', 'silence'):
        p = m(wav[k]); got[k] = p.label
        say(f'{k:8s}', [(d['label'], round(d['score'], 3)) for d in p.preds[:2]])
        assert p.score > 0.2, (k, p.score)         # a sigmoid head's own score, not a softmax over 521
    assert got['sine'] == 'Sine wave' and got['silence'] == 'Silence', got
    assert 'oise' in got['noise'] or 'tatic' in got['noise'], got

# %% ------------------------------------------------------------------ the layer above

@case
def tasks_and_tools(im):
    'One line per job, and the same jobs shaped for a model to call.'
    ps = classify(PICS, IMAGENET, topk=1)
    s = summarize(ps); say('classify folder', s.n, 'items,', s.failed, 'failed, mean', s.mean_score)
    assert s.n == len(PHOTOS) and s.failed == 0, s

    inside = PICS/'sorted'                     # inside the folder being read, which is the usual case
    r = sort_images(PICS, IMAGENET, dest=inside, min_score=0.3, how='link', dry_run=False)
    say('sorted', r.moved, 'into', len(r.labels), 'folders:', sorted(x[:18] for x in r.labels))
    assert r.moved == len(PHOTOS) and any('tabby' in x for x in r.labels), r
    again = sort_images(PICS, IMAGENET, dest=inside, dry_run=True)
    assert again.n == len(PHOTOS), again.n     # exclude= keeps the second run out of the first run's output

    d = classify_image(str(im.cats), IMAGENET, topk=2)
    say('classify_image', [(x['label'][:12], round(x['score'], 3)) for x in d['labels']])
    assert len(d['labels']) == 2 and d['error'] is None, d       # topk survived as an int

    d = detect_image(str(im.bus), 'webnn/yolov8n', conf=0.8, limit=3)
    say('detect_image', d['n'], 'over conf=0.8')
    assert 0 < d['n'] <= 6, d                                   # conf survived as a float

    d = segment_image(str(im.bus), 'Xenova/segformer-b0-finetuned-ade-512-512')
    assert d['classes'] and d['error'] is None, d
    say('segment_image', len(d['classes']), 'classes')

@case
def video(im):
    'Frames out of a real mp4, each with its timestamp.'
    try: import av
    except ImportError: return say('skipped: no av')
    from PIL import Image
    clip = CACHE/'clip.mp4'
    if not clip.exists():
        with av.open(str(clip), 'w') as c:
            st = c.add_stream('libx264', rate=4); st.width, st.height, st.pix_fmt = 640, 480, 'yuv420p'
            for k in ('cats', 'bus', 'bear', 'room'):
                a = np.asarray(Image.open(im[k]).convert('RGB').resize((640, 480)))
                for _ in range(6):
                    for pk in st.encode(av.VideoFrame.from_ndarray(a, format='rgb24')): c.mux(pk)
            for pk in st.encode(): c.mux(pk)
    d = label_video(str(clip), 'webnn/yolov8n', every=1.5, max_frames=5)
    say('frames', [(f['t'], f['label']) for f in d['frames']])
    assert [f['label'] for f in d['frames']][:2] == ['cat', 'person'], d['frames']

# %% ------------------------------------------------------------------ main

def main(only=None) -> int:
    im = photos()
    cases = [f for f in CASES if not only or any(o in f.__name__ for o in only)]
    bad = []
    for f in cases:
        print(f'== {f.__name__}: {f.__doc__.splitlines()[0]}')
        try: f(im)
        except Exception as e:
            bad.append(f.__name__); print(f'    FAILED {type(e).__name__}: {e}')
    print(f'\n{len(cases)-len(bad)}/{len(cases)} passed' + (f', failed: {", ".join(bad)}' if bad else ''))
    return 1 if bad else 0

if __name__ == '__main__': sys.exit(main(sys.argv[1:]))
