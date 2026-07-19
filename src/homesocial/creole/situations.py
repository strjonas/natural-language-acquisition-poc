"""Speech-act schema for the caregiver.

A ``Situation`` is a grounded speech act with slot values drawn from the
island ontology. The full situation space is enumerable, so the utterance
bank has complete coverage by construction. Every act carries:

- hand templates (guaranteed valid, at least three per situation) that act as
  the generation fallback,
- required-token constraints that keep LLM-generated variants semantically
  actionable (a label must reveal the kind, a warning must warn, an answer
  about where must name the kind and the place).
"""

from __future__ import annotations

from dataclasses import dataclass

from homesocial.creole.vocab import UtteranceError, validate_utterance

# The kinds each surface can plausibly be. Kinds are assigned per world, so
# perception alone can never resolve them; the caregiver's label is the only
# source. Revealing kind is the caregiver's main epistemic contribution.
SURFACE_KINDS: dict[str, tuple[str, ...]] = {
    "water": ("food", "water", "danger"),
    "spring": ("food", "water", "danger"),
    "berry": ("food", "water", "danger"),
    "roots": ("food", "water", "danger"),
    "mushroom": ("food", "water", "danger"),
    "hut": ("shelter",),
    "thorn": ("danger",),
    "tree": ("tree",),
    "rock": ("rock",),
}

RESOURCE_KINDS = ("water", "food", "shelter")
CARDINALS = ("north", "south", "east", "west")
DEICTICS = ("here", "there")
LANDMARKS = ("tree", "rock", "hut", "water")
NEEDS = ("hungry", "thirsty", "tired", "hurt")
CORRECTABLE_VERBS = ("eat", "drink", "go", "take")

ACTS = (
    "label",
    "warn",
    "answer_where",
    "ask_state",
    "offer",
    "confirm",
    "deny",
    "praise",
    "correct",
)

ACT_SEMANTICS: dict[str, str] = {
    "label": (
        "Joint attention naming: the caregiver and learner attend to one object; "
        "the caregiver names what it is for the body (its kind)."
    ),
    "warn": (
        "The caregiver warns the learner about a nearby hazard, optionally saying "
        "where it is, before the learner gets hurt."
    ),
    "answer_where": (
        "The learner asked where to find a resource; the caregiver answers with a "
        "direction or a landmark."
    ),
    "ask_state": (
        "The caregiver asks the learner how it feels, either in general or about "
        "one specific bodily need."
    ),
    "offer": (
        "The caregiver offers or hands the learner a resource it can use right now."
    ),
    "confirm": "The caregiver affirms what the learner just did or said.",
    "deny": "The caregiver negates what the learner just did or said.",
    "praise": "The caregiver praises the learner after a good action.",
    "correct": (
        "The caregiver stops the learner from performing a harmful action it is "
        "about to take."
    ),
}


@dataclass(frozen=True)
class Situation:
    act: str
    slots: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.act not in ACTS:
            raise ValueError(f"Unknown speech act: {self.act}.")

    def slot(self, name: str) -> str | None:
        for key, value in self.slots:
            if key == name:
                return value
        return None

    def key(self) -> str:
        parts = [self.act]
        parts.extend(f"{name}={value}" for name, value in self.slots)
        return "|".join(parts)

    @staticmethod
    def from_key(key: str) -> "Situation":
        parts = key.split("|")
        slots = []
        for part in parts[1:]:
            name, _, value = part.partition("=")
            if not value:
                raise ValueError(f"Malformed situation key: {key!r}")
            slots.append((name, value))
        return Situation(act=parts[0], slots=tuple(slots))


def enumerate_situations() -> tuple[Situation, ...]:
    situations: list[Situation] = []
    for surface, kinds in SURFACE_KINDS.items():
        for kind in kinds:
            situations.append(
                Situation("label", (("surface", surface), ("kind", kind)))
            )
    situations.append(Situation("warn"))
    for place in CARDINALS + DEICTICS:
        situations.append(Situation("warn", (("place", place),)))
    for kind in RESOURCE_KINDS:
        for place in CARDINALS + DEICTICS:
            situations.append(
                Situation("answer_where", (("kind", kind), ("place", place)))
            )
        for landmark in LANDMARKS:
            if landmark == kind or (kind == "shelter" and landmark == "hut"):
                continue
            situations.append(
                Situation("answer_where", (("kind", kind), ("landmark", landmark)))
            )
    situations.append(Situation("ask_state"))
    for need in NEEDS:
        situations.append(Situation("ask_state", (("need", need),)))
    for kind in RESOURCE_KINDS:
        situations.append(Situation("offer", (("kind", kind),)))
    situations.append(Situation("confirm"))
    situations.append(Situation("deny"))
    situations.append(Situation("praise"))
    for verb in CORRECTABLE_VERBS:
        situations.append(Situation("correct", (("verb", verb),)))
    return tuple(situations)


def required_tokens(situation: Situation) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(required_all, required_any)`` word sets for valid variants.

    ``required_any`` may be empty, meaning no disjunctive requirement.
    """

    act = situation.act
    if act == "label":
        kind = situation.slot("kind")
        assert kind is not None
        return frozenset({kind}), frozenset()
    if act == "warn":
        place = situation.slot("place")
        required_all = frozenset({place}) if place else frozenset()
        return required_all, frozenset({"danger", "careful", "not"})
    if act == "answer_where":
        kind = situation.slot("kind")
        assert kind is not None
        landmark = situation.slot("landmark")
        if landmark is not None:
            return frozenset({kind, "near", landmark}), frozenset()
        place = situation.slot("place")
        assert place is not None
        return frozenset({kind, place}), frozenset()
    if act == "ask_state":
        need = situation.slot("need")
        if need is None:
            return frozenset({"you"}), frozenset({"what", "how", "feel", "good"})
        return frozenset({"you", need}), frozenset()
    if act == "offer":
        kind = situation.slot("kind")
        assert kind is not None
        return frozenset({kind}), frozenset({"take", "give", "here", "this", "have"})
    if act == "confirm":
        return frozenset({"yes"}), frozenset()
    if act == "deny":
        return frozenset({"no"}), frozenset()
    if act == "praise":
        return frozenset({"good"}), frozenset()
    if act == "correct":
        verb = situation.slot("verb")
        assert verb is not None
        return frozenset({verb}), frozenset({"no", "not"})
    raise ValueError(f"Unknown speech act: {act}.")


def template_variants(situation: Situation) -> tuple[tuple[str, ...], ...]:
    """Hand-written fallback variants; each is validated before returning."""

    act = situation.act
    texts: list[str]
    if act == "label":
        surface = situation.slot("surface")
        kind = situation.slot("kind")
        assert surface is not None and kind is not None
        if surface == kind:
            texts = [f"this {surface}", f"see {surface}", f"{surface} here"]
        else:
            texts = [f"{surface} {kind}", f"this {surface} {kind}", f"this {kind}"]
    elif act == "warn":
        place = situation.slot("place")
        if place is None:
            texts = ["careful danger", "careful", "danger"]
        else:
            texts = [f"danger {place}", f"careful danger {place}", f"not go {place}"]
    elif act == "answer_where":
        kind = situation.slot("kind")
        assert kind is not None
        landmark = situation.slot("landmark")
        if landmark is not None:
            texts = [
                f"{kind} near {landmark}",
                f"find {kind} near {landmark}",
                f"go near {landmark} find {kind}",
            ]
        else:
            place = situation.slot("place")
            assert place is not None
            if place in DEICTICS:
                texts = [f"{kind} {place}", f"see {kind} {place}", f"{kind} {place} now"]
            else:
                texts = [f"{kind} {place}", f"go {place} find {kind}", f"{kind} far {place}"]
    elif act == "ask_state":
        need = situation.slot("need")
        if need is None:
            texts = ["what you feel", "how you feel", "you good"]
        else:
            texts = [f"you {need}", f"you {need} now", f"you feel {need}"]
    elif act == "offer":
        kind = situation.slot("kind")
        assert kind is not None
        texts = [f"take {kind}", f"me give {kind}", f"here {kind}"]
    elif act == "confirm":
        texts = ["yes", "yes good", "yes this good"]
    elif act == "deny":
        texts = ["no", "no bad", "no not this"]
    elif act == "praise":
        texts = ["good", "good you", "this good"]
    elif act == "correct":
        verb = situation.slot("verb")
        assert verb is not None
        texts = [f"no {verb}", f"not {verb} this", f"no {verb} that"]
    else:
        raise ValueError(f"Unknown speech act: {act}.")
    variants = tuple(validate_utterance(text) for text in texts)
    required_all, required_any = required_tokens(situation)
    for words in variants:
        word_set = set(words)
        if not required_all <= word_set:
            raise UtteranceError(
                f"Template {words!r} for {situation.key()!r} misses {required_all}."
            )
        if required_any and not (required_any & word_set):
            raise UtteranceError(
                f"Template {words!r} for {situation.key()!r} misses any of {required_any}."
            )
    return variants


def situation_gloss(situation: Situation) -> str:
    """A natural-language description of the situation, for LLM prompting."""

    act = situation.act
    base = ACT_SEMANTICS[act]
    details: list[str] = []
    surface = situation.slot("surface")
    kind_slot = situation.slot("kind")
    if surface is not None:
        kind = kind_slot or "unknown"
        if surface == kind:
            details.append(f"The object is a {surface}; its name is also its kind.")
        else:
            details.append(
                f"The object looks like '{surface}' and, in this world, its bodily "
                f"kind is '{kind}'. The utterance must reveal the kind '{kind}'."
            )
    elif kind_slot is not None:
        details.append(f"The resource asked about is '{kind_slot}'.")
    place = situation.slot("place")
    if place is not None:
        details.append(f"The location is '{place}'.")
    landmark = situation.slot("landmark")
    if landmark is not None:
        details.append(f"It is near the landmark '{landmark}'.")
    need = situation.slot("need")
    if need is not None:
        details.append(f"The need asked about is '{need}'.")
    verb = situation.slot("verb")
    if verb is not None:
        details.append(f"The harmful action to stop is '{verb}'.")
    return " ".join([base, *details])
