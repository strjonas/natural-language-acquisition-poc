"""Offline utterance-bank generation via OpenRouter.

This is world-content generation, not agent cognition: the LLM proposes
surface variants for each grounded situation, a strict validator rejects
anything outside the closed vocabulary or missing required tokens, and hand
templates guarantee minimum coverage even if every LLM variant is rejected.
Responses are cached on disk so regeneration is incremental and cheap.

Run:

    PYTHONPATH=src python3 -m homesocial.creole.generate_bank \
        --output src/homesocial/creole/data/utterance_bank.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from homesocial.creole.bank import UtteranceBank
from homesocial.creole.situations import (
    Situation,
    enumerate_situations,
    required_tokens,
    situation_gloss,
    template_variants,
)
from homesocial.creole.vocab import CONTENT_WORDS, UtteranceError, validate_utterance

PROMPT_VERSION = "v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_CACHE_DIR = "runs/organism/bank_cache"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            name, _, value = line.strip().partition("=")
            if name == "OPENROUTER_API_KEY" and value:
                return value
    raise RuntimeError(
        "OPENROUTER_API_KEY not found in environment or repository .env file."
    )


def openrouter_chat(
    prompt: str,
    *,
    api_key: str,
    model: str,
    max_tokens: int = 400,
    temperature: float = 0.9,
    timeout: float = 90.0,
) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "reasoning": {"enabled": False},
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"].get("content") or ""
    cost = float(body.get("usage", {}).get("cost", 0.0))
    return content, cost


def build_prompt(situation: Situation, n_variants: int) -> str:
    required_all, required_any = required_tokens(situation)
    templates = [" ".join(words) for words in template_variants(situation)]
    lines = [
        "You are writing utterances for a caregiver character in a tiny island",
        "world with an invented creole language. The language has a small fixed",
        "vocabulary and utterances of at most 5 words. Word order is flexible",
        "and there is no grammar beyond word choice; keep utterances simple and",
        "caregiver-like (as if speaking to a young child).",
        "",
        "Allowed words (use ONLY these, lowercase, no punctuation):",
        ", ".join(CONTENT_WORDS),
        "",
        f"Speech situation: {situation_gloss(situation)}",
    ]
    if required_all:
        lines.append(
            "Every utterance MUST contain the word(s): "
            + ", ".join(sorted(required_all))
        )
    if required_any:
        lines.append(
            "Every utterance MUST contain at least one of: "
            + ", ".join(sorted(required_any))
        )
    lines.extend(
        [
            "",
            "Example utterances for this situation: " + "; ".join(templates),
            "",
            f"Write {n_variants} DISTINCT utterances for this situation, varied in",
            "wording and length (1 to 5 words each). Output ONLY a JSON array of",
            'strings, e.g. ["water here", "go north find water"]. No other text.',
        ]
    )
    return "\n".join(lines)


def parse_variant_texts(content: str) -> list[str]:
    text = content.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    return [line.strip().strip('",') for line in text.splitlines() if line.strip()]


def filter_valid(
    situation: Situation, texts: list[str]
) -> tuple[list[tuple[str, ...]], int]:
    required_all, required_any = required_tokens(situation)
    valid: list[tuple[str, ...]] = []
    rejected = 0
    seen: set[tuple[str, ...]] = set()
    for text in texts:
        try:
            words = validate_utterance(text)
        except UtteranceError:
            rejected += 1
            continue
        word_set = set(words)
        if not required_all <= word_set or (required_any and not (required_any & word_set)):
            rejected += 1
            continue
        if words in seen:
            continue
        seen.add(words)
        valid.append(words)
    return valid, rejected


class BankGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        cache_dir: Path,
        variants_per_key: int,
        max_attempts: int = 2,
    ):
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self.variants_per_key = variants_per_key
        self.max_attempts = max_attempts
        self.total_cost = 0.0
        self.total_rejected = 0

    def _cache_path(self, situation: Situation, attempt: int) -> Path:
        digest = hashlib.sha256(
            f"{PROMPT_VERSION}|{self.model}|{situation.key()}|{attempt}".encode()
        ).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _fetch_texts(self, situation: Situation, attempt: int) -> list[str]:
        cache_path = self._cache_path(situation, attempt)
        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            return list(record["texts"])
        prompt = build_prompt(situation, self.variants_per_key)
        if attempt > 0:
            prompt += (
                "\nMake these utterances different from earlier ones; explore other"
                " allowed words."
            )
        content, cost = openrouter_chat(
            prompt,
            api_key=self.api_key,
            model=self.model,
            temperature=0.9 + 0.1 * attempt,
        )
        self.total_cost += cost
        texts = parse_variant_texts(content)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"situation": situation.key(), "texts": texts, "cost": cost},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return texts

    def variants_for(self, situation: Situation) -> tuple[tuple[str, ...], ...]:
        collected: list[tuple[str, ...]] = list(template_variants(situation))
        seen = set(collected)
        for attempt in range(self.max_attempts):
            try:
                texts = self._fetch_texts(situation, attempt)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
                print(f"  fetch failed for {situation.key()!r} (attempt {attempt}): {error}")
                continue
            valid, rejected = filter_valid(situation, texts)
            self.total_rejected += rejected
            for words in valid:
                if words not in seen:
                    seen.add(words)
                    collected.append(words)
            if len(collected) >= self.variants_per_key:
                break
        return tuple(collected)


def main() -> None:
    args = _parse_args()
    situations = enumerate_situations()
    if args.max_keys is not None:
        situations = situations[: args.max_keys]
    if args.dry_run:
        variants_by_key = {
            situation.key(): template_variants(situation) for situation in situations
        }
        bank = UtteranceBank(variants_by_key)
    else:
        generator = BankGenerator(
            api_key=load_api_key(),
            model=args.model,
            cache_dir=Path(args.cache_dir),
            variants_per_key=args.variants_per_key,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(generator.variants_for, situations))
        variants_by_key = {
            situation.key(): variants
            for situation, variants in zip(situations, results)
        }
        bank = UtteranceBank(variants_by_key)
        print(
            f"generation cost ${generator.total_cost:.4f}, "
            f"rejected {generator.total_rejected} variants"
        )
    if args.max_keys is None:
        bank.validate()
    bank.save(args.output)
    counts = [len(bank.variants(situation)) for situation in situations]
    print(
        f"saved {len(situations)} situations to {args.output}; "
        f"variants per key min/mean/max = "
        f"{min(counts)}/{sum(counts) / len(counts):.1f}/{max(counts)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="src/homesocial/creole/data/utterance_bank.jsonl"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--variants-per-key", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-keys", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.variants_per_key < 1:
        parser.error("--variants-per-key must be positive")
    return args


if __name__ == "__main__":
    main()
