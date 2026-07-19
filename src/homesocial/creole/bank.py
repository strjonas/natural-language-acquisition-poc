"""Utterance bank: cached surface variants for every situation key.

The bank is generated offline (see ``generate_bank.py``) and loaded read-only
at environment construction. Sampling is seed-deterministic. Besides grounded
sampling by exact situation, the bank supports per-event sampling from a whole
speech-act family for controls that preserve the act while destroying its
grounded slot information.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from homesocial.creole.situations import (
    Situation,
    enumerate_situations,
    required_tokens,
)
from homesocial.creole.vocab import UtteranceError, validate_utterance

MIN_VARIANTS_PER_KEY = 3


class UtteranceBank:
    def __init__(self, variants_by_key: dict[str, tuple[tuple[str, ...], ...]]):
        self._variants_by_key = dict(variants_by_key)
        variants_by_act: dict[str, list[tuple[str, ...]]] = {}
        for key, variants in self._variants_by_key.items():
            act = Situation.from_key(key).act
            variants_by_act.setdefault(act, []).extend(variants)
        self._variants_by_act = {
            act: tuple(variants) for act, variants in variants_by_act.items()
        }

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._variants_by_key)

    def variants(self, situation: Situation) -> tuple[tuple[str, ...], ...]:
        key = situation.key()
        if key not in self._variants_by_key:
            raise KeyError(f"No variants for situation {key!r}.")
        return self._variants_by_key[key]

    def sample(self, situation: Situation, rng: random.Random) -> tuple[str, ...]:
        variants = self.variants(situation)
        return variants[rng.randrange(len(variants))]

    def sample_from_act(self, act: str, rng: random.Random) -> tuple[str, ...]:
        """Sample a variant from ``act`` independently of grounded slots."""

        if act not in self._variants_by_act:
            raise KeyError(f"No variants for speech act {act!r}.")
        variants = self._variants_by_act[act]
        return variants[rng.randrange(len(variants))]

    def validate(self, *, check_required: bool = True) -> None:
        expected = {situation.key(): situation for situation in enumerate_situations()}
        missing = sorted(set(expected) - set(self._variants_by_key))
        if missing:
            raise UtteranceError(f"Bank is missing situation keys: {missing[:5]} ...")
        for key, variants in self._variants_by_key.items():
            if len(variants) < MIN_VARIANTS_PER_KEY:
                raise UtteranceError(
                    f"Situation {key!r} has {len(variants)} variants; "
                    f"minimum is {MIN_VARIANTS_PER_KEY}."
                )
            situation = expected.get(key)
            for words in variants:
                validated = validate_utterance(" ".join(words))
                if validated != tuple(words):
                    raise UtteranceError(f"Non-canonical variant {words!r} in {key!r}.")
                if check_required and situation is not None:
                    required_all, required_any = required_tokens(situation)
                    word_set = set(words)
                    if not required_all <= word_set:
                        raise UtteranceError(
                            f"Variant {words!r} for {key!r} misses {sorted(required_all)}."
                        )
                    if required_any and not (required_any & word_set):
                        raise UtteranceError(
                            f"Variant {words!r} for {key!r} misses any of "
                            f"{sorted(required_any)}."
                        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for key in sorted(self._variants_by_key):
                record = {
                    "situation": key,
                    "variants": [" ".join(words) for words in self._variants_by_key[key]],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def load(path: str | Path, *, validate: bool = True) -> "UtteranceBank":
        variants_by_key: dict[str, tuple[tuple[str, ...], ...]] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = record["situation"]
                Situation.from_key(key)
                variants_by_key[key] = tuple(
                    tuple(text.split()) for text in record["variants"]
                )
        bank = UtteranceBank(variants_by_key)
        if validate:
            bank.validate()
        return bank


def default_bank_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "utterance_bank.jsonl"


def load_default_bank() -> UtteranceBank:
    return UtteranceBank.load(default_bank_path())
