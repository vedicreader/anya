"""What does fitting buy over a pattern bank, and what does it cost.

`vishalakshi.pii` is 34 kinds with a checksum each and 1.000 out-of-sample precision on its own
corpus. It cannot know `EMP-483920`, and a fitted model cannot know a checksum. So the question is
not whether either is good. It is what the two of them together should do when they disagree.

Five arbitrations on the same splits of `evals/mkpii.py`:

  baseline   the pattern bank alone, which is what `fit` gives you back with nothing to fit on
  learned    the fitted tagger alone
  defer      the tagger only for kinds the bank never produced (the default)
  union      both, the longer span winning an overlap
  hybrid     both, the tagger allowed to drop a kind it was trained on

Run it twice. With `vishalakshi` installed the baseline is the real bank; with `--floor` it is anya's
six-pattern fallback, which is what a user who installed neither has. The two answers differ, and
which mode to use follows from which baseline you have.

Splits hold out whole templates, so nothing is scored on a sentence form it was fitted on. Metrics
are the mean over `--seeds` splits. The intervals are a paired bootstrap over the pooled held-out
documents of every split, not over one split, because one split of 240 examples does not settle it.

    python evals/pii_learned.py [--seeds 5] [--floor]
"""
import argparse, os, statistics, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anya.pii import (PiiDetector, baseline, bootstrap, counts_of, dataset, evaluate, fit, prf)
from evals.mkpii import corpus

MODES = ('baseline', 'learned', 'defer', 'union', 'hybrid')
mean = lambda xs: statistics.mean(xs) if xs else 0.0


def systems(tr, base):
    "One detector per arbitration, all fitted on the same training half."
    det = {m: fit(tr, mode=m, base=base, seed=0) for m in MODES if m != 'baseline'}
    return {'baseline': PiiDetector(mode='baseline', base=base), **det}


def run(seeds=5, n=240, valid=0.25):
    base = baseline()
    ds = dataset(corpus(n, seed=0))
    print(f'{len(ds)} examples, {sum(len(e.text) for e in ds):,} characters, {len(ds.kinds)} kinds, '
          f'{ds.counts()["groups"]} templates, baseline={base.__name__}, {seeds} splits\n')

    agg, pooled, kind_r, kind_fp = {}, {}, {}, {}
    for seed in range(seeds):
        tr, va = ds.split(valid, seed=seed)
        for nm, det in systems(tr, base).items():
            m = evaluate(det, va, max_errors=0)
            d = agg.setdefault(nm, {k: [] for k in ('sp', 'sr', 'sf', 'df', 'ca', 'ms')})
            d['sp'].append(m.spans.precision); d['sr'].append(m.spans.recall); d['sf'].append(m.spans.f1)
            d['df'].append(m.doc['f1']); d['ms'].append(m.ms_per_doc)
            if m.class_accuracy is not None: d['ca'].append(m.class_accuracy)
            pooled.setdefault(nm, []).append(counts_of(m.per_doc, 'spans'))
            for k, v in m.by_kind.items():
                if v['support']: kind_r.setdefault(nm, {}).setdefault(k, []).append(v['recall'])
                kind_fp.setdefault(nm, {})[k] = kind_fp.setdefault(nm, {}).get(k, 0) + v['fp']

    print(f'{"arbitration":<12}{"span P":>8}{"span R":>8}{"span F1":>9}{"doc F1":>8}'
          f'{"invented":>10}{"ms/doc":>9}')
    for nm in MODES:
        d, fp = agg[nm], sum(kind_fp.get(nm, {}).values())
        print(f'{nm:<12}{mean(d["sp"]):>8.3f}{mean(d["sr"]):>8.3f}{mean(d["sf"]):>9.3f}'
              f'{mean(d["df"]):>8.3f}{fp:>10d}{mean(d["ms"]):>9.2f}')
    print(f'\n"invented" is every false-positive span over all {seeds} splits. The document classifier\'s '
          f'own class accuracy was {mean(agg["defer"]["ca"]):.3f}; it does not decide `has_pii` while a '
          'tagger is fitted.')
    print(f'span F1 by split, defer: ' + ' '.join(f'{x:.3f}' for x in agg['defer']['sf']))

    print(f'\nspan F1 against the baseline, paired bootstrap over {sum(len(a) for a in pooled["baseline"])} '
          'pooled held-out documents')
    A = np.vstack(pooled['baseline'])
    for nm in MODES[1:]:
        c = bootstrap(A, np.vstack(pooled[nm]), n=2000, seed=0)
        print(f'  {nm:<10}{c.delta:+.3f}  [{c.lo:+.3f}, {c.hi:+.3f}]  '
              f'{"differs" if c.difference else "no difference"}')

    kinds = sorted({k for v in kind_r.values() for k in v})
    print('\nrecall by kind, mean over the splits that held any')
    print(f'{"":<12}' + ''.join(f'{k[:11]:>12}' for k in kinds))
    for nm in MODES:
        v = kind_r.get(nm, {})
        print(f'{nm:<12}' + ''.join(f'{mean(v[k]):>12.2f}' if v.get(k) else f'{"-":>12}' for k in kinds))

    print('\nfalse positives by kind, all splits')
    print(f'{"":<12}' + ''.join(f'{k[:11]:>12}' for k in kinds))
    for nm in MODES:
        v = kind_fp.get(nm, {})
        print(f'{nm:<12}' + ''.join(f'{v.get(k, 0):>12d}' for k in kinds))

    tr, va = ds.split(valid, seed=0)
    empty, pat = fit([], base=base), PiiDetector(mode='baseline', base=base)
    same = all(empty.spans(e.text) == pat.spans(e.text) for e in va)
    print(f'\nnothing to fit on: {len(va)} documents answered identically to the baseline: {same}')
    print(f'  fit([]) is {empty!r}')

    print('\nablations on split 0, span F1')
    for label, kw in (('as shipped', {}),
                      ('no baseline features', dict(base_feats=False)),
                      ('no context window', dict(window=0)),
                      ('unweighted classes', dict(class_weight=None)),
                      ('C=1 not 4', dict(C=1.0)),
                      ('window=4 not 2', dict(window=4))):
        m = evaluate(fit(tr, base=base, seed=0, **kw), va, max_errors=0)
        print(f'  {label:<22}{m.spans.f1:.3f}  (P {m.spans.precision:.3f} R {m.spans.recall:.3f})')
    return agg


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--n', type=int, default=240)
    ap.add_argument('--floor', action='store_true', help="force anya's own floor as the baseline")
    a = ap.parse_args()
    if a.floor: os.environ['ANYA_PII_BASELINE'] = 'floor'
    run(a.seeds, a.n)
