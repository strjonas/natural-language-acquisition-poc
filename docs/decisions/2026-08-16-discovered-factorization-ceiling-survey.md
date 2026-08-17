# Ceiling survey: probe66, is there anything to discover about the factorization?

Date: 2026-08-16
Status: SURVEY. Instruments, not a result. No gate is scored here. This is the
measurement that makes probe66 well-posed, and it exists because the obvious
design for probe66 turned out to be the wrong one.

Lever: `tangled_size_words` on `ReportConfig`, default off.
Guards: `tests/test_portion_request.py`, three tests at the end of the file.
Seed bands: `1,010,000,000`, `1,010,500,000`, `1,011,000,000` -- probe65's
construction band, disjoint from every treatment band in use.

**Superseded in one respect and only one.** Section 5 below says nothing had been
built or run; that was true when it was written and is no longer. The learner,
its preregistration and a five-seed treatment all landed the same day -- see
`2026-08-16-discovered-convention-preregistration.md` and
`2026-08-16-discovered-convention-result.md` (6/6 gates). Every measurement in
section 3 stands and is what the gates were locked against. The record is left
as it was written.

## 1. What probe64 left open

Probe64 moved property 4, *productive under novel demand*, from **no** to
**partial**, and stated the one reason it is not **yes**:

> That the listener-model factorization is discovered. It is given.

Its `factored` listener holds one belief per size word, pooled over needs -- *this
word means large, whatever I am asking for* -- and that pooling is what lets it
utter a combination it has never uttered. But the designer chose to pool, and the
world happened to reward pooling. If the credit belongs to the organism, the
organism has to **find out** whether the size word's meaning transfers.

`md/archive/DIRECTION_2026-07-26.md` and probe61's preregistration both make the
same demand of any discovery claim: **the ground truth has to move.** A learner
that always answers "it factors" must be told apart from one that finds out.

## 2. The lever

`tangled_size_words`, default off. Off, the caregiver reads "more" as large for
every need -- the species convention every earlier probe used, and the draw still
happens so no other stream moves. On, it draws its own word-to-size orientation
**per need** at birth, so "more" may be large for food and small for water. The
convention is still one deterministic fact per life and still learnable from the
surface that appears beside each grant; what changes is that it no longer
factors.

Three tests guard it: that a life with the lever off is tick-for-tick identical
to one that never mentions it, that both orientations actually occur when it is
on, and that a tangled caregiver grants by the *pair* rather than by the word.

## 3. The measurement, and it overturned the design it was run to check

The obvious expectation -- the one this survey was run to confirm -- is that a
pooling model *degrades gracefully* in a tangled world. Pooling "more" across
three needs whose orientations disagree should give about 0.5, which is an honest
"I do not know", and a model that says so is not doing anything wrong. If that
were true, probe64's given factorization would need no discovery at all and
probe66 would have to be built around partial evidence on the held-out cell.

It is not true. Probe64's `composition_holdout` run unchanged in both worlds,
40 lives on each of three disjoint seed bands:

| world | `factored` first-use | `tabular` first-use |
|---|---:|---:|
| factoring (`tangled_size_words` off) | 0.988 / 1.000 / 1.000 | 0.500 / 0.500 / 0.500 |
| tangled (`tangled_size_words` on) | **0.350 / 0.375 / 0.500** | 0.500 / 0.500 / 0.500 |

**The given factorization is near-perfect when the world factors and at or below
chance when it does not** -- mean 0.408 against the 0.500 a model with no
evidence scores. It does not shrug; it commits, and in the tangled world it
commits wrongly. `tabular` is exactly 0.500 in all six cells, as it must be: it
has two empty cells and nothing to fill them from, in either world.

That is the discriminating control a discovery claim needs, and it is now
measured rather than assumed. It also means probe66 can be built on the **zero
evidence** holdout probe64 already uses, with no partial-evidence complication.

## 4. What this fixes about probe66's design

The experiment is now well-posed with three listener models on identical
transcripts in two worlds whose true structure differs:

- `factored` -- probe64's, pools always. Should be ~1.00 factoring, ~0.41 tangled.
- `tabular` -- probe64's, never pools. Exactly 0.500 in both.
- `discovered` -- learns how much the word's meaning transfers, from the same
  evidence, and is allowed to **decline**: to say the need word with no size word
  at all, which is what every organism before probe64 did and which probe64
  measured as survivable (`no_size` survives 0.630 against `always_large`'s
  0.005). Declining is therefore a real move in this ecology and not a scoring
  dodge.

The claim probe66 would put up to be attacked:

> An organism can find out **whether a word means the same thing whatever it is
> asking for**, from the same evidence that teaches it what the word does, and
> can act on the difference -- generalizing where generalization is licensed and
> declining where it is not -- while a model handed the factorization
> generalizes in both and is confidently wrong in one.

Gates would then be: the discovered model matches `factored` in the factoring
world and beats it decisively in the tangled one; its recovered pooling strength
is materially higher in the factoring world than the tangled one, per seed
(the moved-ground-truth gate); `factored` fails in the tangled world (measured
here at 0.408); `tabular` sits at exactly 0.500 in both; and with the lever off
everything reproduces probe64.

Because declining is scored, the gate has to name what a decline counts as.
The honest form is two numbers, not one: **how often it speaks** and **how often
it is right when it speaks**, so that a model which declines everywhere cannot
pass by never being wrong.

## 5. What had not been run when this was written

Everything in section 4. At the time of writing no `discovered` learner existed,
no gate was locked, and no treatment seed had been spent; this document recorded
only the lever, its guards, and the measurement in section 3. All of section 4
was subsequently built and run -- see the header. The measurement in section 3
remains what the gates were locked against, and it is still the reason the
experiment was worth building: it came out the opposite way from the reasoning it
was run to check.
