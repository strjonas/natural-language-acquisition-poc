"""Closed island-creole vocabulary.

The vocabulary is deliberately small and fixed: the agent must be able to
acquire it from thousands of in-loop exposures, and the bank validator must be
able to reject any generated utterance that steps outside it. Token ids are
stable across runs because the lists below are ordered.
"""

from __future__ import annotations

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = (PAD_TOKEN, EOS_TOKEN, UNK_TOKEN)

# Surface names of concrete objects on the island (what things look like).
SURFACE_WORDS = (
    "water",
    "spring",
    "berry",
    "roots",
    "mushroom",
    "hut",
    "thorn",
    "tree",
    "rock",
)

# Functional kinds (what things do to the body). Hidden from perception in
# language-necessary worlds; revealing them is the teacher's main job.
KIND_WORDS = (
    "food",
    "shelter",
    "danger",
)

PLACE_WORDS = (
    "north",
    "south",
    "east",
    "west",
    "here",
    "there",
    "near",
    "far",
)

NEED_WORDS = (
    "hungry",
    "thirsty",
    "tired",
    "hurt",
    "safe",
)

VERB_WORDS = (
    "go",
    "come",
    "take",
    "eat",
    "drink",
    "rest",
    "look",
    "see",
    "avoid",
    "want",
    "need",
    "give",
    "help",
    "feel",
    "find",
    "wait",
    "ask",
)

SOCIAL_WORDS = (
    "you",
    "me",
    "yes",
    "no",
    "not",
    "what",
    "where",
    "how",
    "this",
    "that",
    "good",
    "bad",
    "now",
    "careful",
    "more",
)

CONTENT_WORDS = (
    SURFACE_WORDS + KIND_WORDS + PLACE_WORDS + NEED_WORDS + VERB_WORDS + SOCIAL_WORDS
)

VOCAB: tuple[str, ...] = SPECIAL_TOKENS + CONTENT_WORDS

TOKEN_TO_ID: dict[str, int] = {token: index for index, token in enumerate(VOCAB)}

# Utterances are at most 5 words; encoded form appends <eos> and pads.
MAX_UTTERANCE_WORDS = 5
TOKENS_PER_UTTERANCE = MAX_UTTERANCE_WORDS + 1


class UtteranceError(ValueError):
    """Raised when an utterance violates the closed-vocabulary contract."""


def tokenize(text: str) -> tuple[str, ...]:
    """Split an utterance into lowercase word tokens without validating them."""

    return tuple(part for part in text.strip().lower().split() if part)


def validate_utterance(text: str) -> tuple[str, ...]:
    """Return the word tokens of ``text`` or raise ``UtteranceError``."""

    words = tokenize(text)
    if not words:
        raise UtteranceError("Utterance is empty.")
    if len(words) > MAX_UTTERANCE_WORDS:
        raise UtteranceError(f"Utterance has {len(words)} words; max is {MAX_UTTERANCE_WORDS}: {text!r}")
    for word in words:
        if word in SPECIAL_TOKENS:
            raise UtteranceError(f"Special token used as word: {word!r}")
        if word not in TOKEN_TO_ID:
            raise UtteranceError(f"Out-of-vocabulary word {word!r} in {text!r}")
    return words


def encode_utterance(words: tuple[str, ...] | None) -> tuple[int, ...]:
    """Encode words as fixed-length token ids: words, <eos>, then <pad>."""

    if words is None:
        return (TOKEN_TO_ID[PAD_TOKEN],) * TOKENS_PER_UTTERANCE
    if len(words) > MAX_UTTERANCE_WORDS:
        raise UtteranceError(f"Cannot encode {len(words)} words; max is {MAX_UTTERANCE_WORDS}.")
    ids = [TOKEN_TO_ID.get(word, TOKEN_TO_ID[UNK_TOKEN]) for word in words]
    ids.append(TOKEN_TO_ID[EOS_TOKEN])
    while len(ids) < TOKENS_PER_UTTERANCE:
        ids.append(TOKEN_TO_ID[PAD_TOKEN])
    return tuple(ids)
