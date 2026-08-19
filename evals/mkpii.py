"""Build the PII dataset anya's tests and evals run against.

An organisation's PII is mostly not the world's PII. Its identifiers have local shapes no pattern
bank ships (`EMP-483920`), and its *non*-identifiers have the same shapes as the ones every pattern
bank does ship (a 16-digit device serial that passes Luhn, an invoice reference shaped exactly like
a patient reference). So the corpus is built from three groups:

  org        five local identifier kinds, the thing a fitted model is supposed to learn
  lookalike  the same shapes carrying no identity, half of them checksum-valid: the veto test
  standard   email, card, ssn, phone: what the pattern bank already gets right and must keep

Every example carries a `group`, which is the template it came from. `Dataset.split` holds groups
out whole, so a fitted model is never scored on a sentence template it was trained on.

    python evals/mkpii.py nbs/fixtures/pii_org.jsonl 240
"""
import json, random, sys
from pathlib import Path


def luhn_number(rng, n=16):
    "A digit run that passes Luhn, built by choosing the check digit rather than by rejection."
    ds = [rng.randrange(10) for _ in range(n - 1)]
    ds[0] = rng.randrange(2, 7)
    tot, parity = 0, n % 2
    for i, d in enumerate(ds):
        if i % 2 == parity: d = d*2 - 9 if d*2 > 9 else d*2
        tot += d
    return ''.join(map(str, ds)) + str((10 - tot % 10) % 10)


def mod11_number(rng):
    "Ten digits that pass the NHS mod-11 check, so the pattern bank has to call it identity."
    while True:
        ds = [rng.randrange(10) for _ in range(9)]
        chk = 11 - sum(d*(10-i) for i, d in enumerate(ds)) % 11
        if chk == 10: continue
        return ''.join(map(str, ds)) + str(0 if chk == 11 else chk)


#: kind -> value builders. Two forms each: the second is a separator the training split never sees
#: when the corpus is split by group, which is what stops the eval measuring memorised punctuation.
ORG_VALUES = {
    'emp_id':      [lambda r: f'EMP-{r.randrange(100000, 999999)}',
                    lambda r: f'EMP/{r.randrange(100000, 999999)}'],
    'patient_ref': [lambda r: f'PT/{r.randrange(2019, 2026)}/{r.randrange(10000, 99999)}',
                    lambda r: f'PT-{r.randrange(2019, 2026)}-{r.randrange(10000, 99999)}'],
    'badge':       [lambda r: f'B-{r.randrange(10000, 99999)}',
                    lambda r: f'BDG{r.randrange(10000, 99999)}'],
    'case_ref':    [lambda r: f'CASE-{r.randrange(2019, 2026)}-{r.randrange(1000, 9999)}',
                    lambda r: f'CASE {r.randrange(2019, 2026)}/{r.randrange(1000, 9999)}'],
    'staff_login': [lambda r: f'{r.choice("abcdefgjkmp")}.{r.choice(["okonkwo","vance","nakamura","halvorsen","patel"])}',
                    lambda r: f'{r.choice(["okonkwo","vance","nakamura","halvorsen","patel"])}{r.randrange(10, 99)}'],
}

#: One sentence per template, `{v}` where the identifier goes. Templates are the split unit.
ORG_TEMPLATES = {
    'emp_id': ['The reviewer of record is {v} and the decision stands.',
               'Escalated by {v} after the second call.',
               'Timesheet for {v} was approved late.',
               'Access was granted to {v} on the same day.'],
    'patient_ref': ['Referral {v} was triaged on arrival.',
                    'Notes against {v} were amended twice.',
                    'The scan booked under {v} did not go ahead.',
                    'Discharge summary for {v} is outstanding.'],
    'badge': ['Entry logged on badge {v} at the loading bay.',
              'Badge {v} was reported lost in March.',
              'The door was held open for {v} by a contractor.',
              'Badge {v} no longer opens the third floor.'],
    'case_ref': ['Case {v} was settled without a hearing.',
                 'The bundle for {v} runs to four hundred pages.',
                 'Counsel instructed on {v} has withdrawn.',
                 'Costs in {v} were assessed on the standard basis.'],
    'staff_login': ['Signed off by {v} in the approvals queue.',
                    'The change was pushed by {v} on Friday.',
                    'Mailbox delegation was given to {v}.',
                    'The ticket was reassigned to {v} overnight.'],
}

#: The same shapes with no identity in them. `serial` and `part` carry valid checksums, so a pattern
#: bank reads them as a payment card and an NHS number: they are the whole reason to fit anything.
LOOKALIKES = {
    'order':  ['Order ORD-{n6} shipped from the Leeds depot.',
               'ORD/{n6} was cancelled before picking.',
               'The pallet against ORD-{n6} arrived short.'],
    'cost':   ['Charged to cost centre CC-{n5} for the quarter.',
               'CC-{n5} is closed to new bookings.',
               'The overspend sits in CC-{n5}.'],
    'asset':  ['Asset ASSET-{n5} was written down in full.',
               'ASSET-{n5} is booked to the Bristol site.',
               'The lease on ASSET-{n5} ends in June.'],
    'invoice':['Invoice INV/{y}/{n5} remains unpaid.',
               'INV-{y}-{n5} was raised against the wrong entity.',
               'The credit note reverses INV/{y}/{n5} in full.'],
    'serial': ['Device serial {luhn} failed the drop test.',
               'The replacement unit ships as serial {luhn}.',
               'Serial {luhn} was returned under warranty.'],
    'part':   ['Part {part} is superseded by the revised bracket.',
               'The bill of materials lists {part} twice.',
               'Stock of {part} runs out in August.'],
}

#: What the pattern bank already finds. A fitted model must not cost the corpus these.
STANDARD = {
    'email':  ['Send the file to {v} before noon.', 'Correspondence goes to {v} from now on.'],
    'card':   ['Payment taken on card {v} at the till.', 'The card ending in the same digits, {v}, was declined.'],
    'ssn':    ['SSN {v} is on file with payroll.', 'The form lists {v} in the wrong box.'],
    'phone':  ['Call {v} before the site closes.', 'The number on the account is {v}.'],
}

STANDARD_VALUES = {
    'email': lambda r: f'{r.choice(["jane","arun","mira","tom"])}.{r.choice(["doe","patel","ross","okonkwo"])}@example.com',
    'card':  lambda r: luhn_number(r),
    'ssn':   lambda r: f'{r.randrange(1, 665):03d}-{r.randrange(1, 99):02d}-{r.randrange(1, 9999):04d}',
    'phone': lambda r: f'+44 20 7946 {r.randrange(1000, 9999)}',
}

#: Prose with nothing in it, so precision has somewhere to fail.
CLEAN = ['The committee deferred the decision to the next quarter.',
         'Conformity with EN 60601-1 was assessed in full.',
         'Delivery is scheduled for the following quarter subject to approval.',
         'The build takes twenty minutes and costs nothing to run again.',
         'Attendance was down on the same week last year.',
         'See the specification and the change log that accompanies it.',
         'Section 12 Place of performance is unchanged.',
         'Figure 4 Way of working, as adopted by the board.']


def _lookalike(rng, cls, tpl):
    "One lookalike sentence, and nothing to mark in it."
    return tpl.format(n6=rng.randrange(100000, 999999), n5=rng.randrange(10000, 99999),
                      y=rng.randrange(2019, 2026), luhn=luhn_number(rng), part=mod11_number(rng))


def corpus(n:int=240, seed:int=0) -> list:
    "`n` examples as dicts of `text`, `spans` and `label`, each tagged with the template it came from."
    rng, out = random.Random(seed), []
    kinds = list(ORG_TEMPLATES)
    while len(out) < n:
        r = rng.random()
        if r < 0.42:                                            # org identifiers
            k = kinds[len(out) % len(kinds)]
            ti = rng.randrange(len(ORG_TEMPLATES[k]))
            val = rng.choice(ORG_VALUES[k])(rng)
            text = ORG_TEMPLATES[k][ti].format(v=val)
            i = text.index(val)
            out.append(dict(text=text, spans=[[i, i + len(val), k]], label='pii', group=f'{k}:{ti}'))
        elif r < 0.68:                                          # the shapes that are not identity
            cls = rng.choice(list(LOOKALIKES))
            ti = rng.randrange(len(LOOKALIKES[cls]))
            out.append(dict(text=_lookalike(rng, cls, LOOKALIKES[cls][ti]), spans=[], label='clean',
                            group=f'neg_{cls}:{ti}'))
        elif r < 0.86:                                          # what the patterns already find
            k = rng.choice(list(STANDARD))
            ti = rng.randrange(len(STANDARD[k]))
            val = STANDARD_VALUES[k](rng)
            text = STANDARD[k][ti].format(v=val)
            i = text.index(val)
            out.append(dict(text=text, spans=[[i, i + len(val), k]], label='pii', group=f'{k}:{ti}'))
        else:                                                   # prose
            ti = rng.randrange(len(CLEAN))
            out.append(dict(text=CLEAN[ti], spans=[], label='clean', group=f'clean:{ti}'))
    return out


def write(dest, n=240, seed=0):
    p = Path(dest); p.parent.mkdir(parents=True, exist_ok=True)
    rows = corpus(n, seed)
    p.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    kinds = {}
    for r in rows:
        for _, _, k in r['spans']: kinds[k] = kinds.get(k, 0) + 1
    print(f'{p}: {len(rows)} examples, {sum(r["label"] == "pii" for r in rows)} positive, '
          f'{len(set(r["group"] for r in rows))} groups')
    print('  spans:', dict(sorted(kinds.items(), key=lambda t: -t[1])))
    return p


if __name__ == '__main__':
    write(sys.argv[1] if len(sys.argv) > 1 else 'nbs/fixtures/pii_org.jsonl',
          int(sys.argv[2]) if len(sys.argv) > 2 else 240)
