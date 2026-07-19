import random
import tempfile
import unittest
from pathlib import Path

from homesocial.creole.bank import UtteranceBank, load_default_bank
from homesocial.creole.generate_bank import build_prompt, filter_valid, parse_variant_texts
from homesocial.creole.situations import (
    SURFACE_KINDS,
    Situation,
    enumerate_situations,
    required_tokens,
    template_variants,
)
from homesocial.creole.vocab import (
    EOS_TOKEN,
    PAD_TOKEN,
    TOKEN_TO_ID,
    TOKENS_PER_UTTERANCE,
    UtteranceError,
    VOCAB,
    encode_utterance,
    validate_utterance,
)


class VocabTest(unittest.TestCase):
    def test_vocab_has_no_duplicates(self):
        self.assertEqual(len(VOCAB), len(set(VOCAB)))

    def test_vocab_words_are_lowercase_single_tokens(self):
        for word in VOCAB[3:]:
            self.assertEqual(word, word.lower())
            self.assertNotIn(" ", word)

    def test_validate_rejects_oov_and_long(self):
        with self.assertRaises(UtteranceError):
            validate_utterance("water banana")
        with self.assertRaises(UtteranceError):
            validate_utterance("go go go go go go")
        with self.assertRaises(UtteranceError):
            validate_utterance("")

    def test_encode_shape_and_padding(self):
        encoded = encode_utterance(("water", "north"))
        self.assertEqual(len(encoded), TOKENS_PER_UTTERANCE)
        self.assertEqual(encoded[2], TOKEN_TO_ID[EOS_TOKEN])
        self.assertEqual(encoded[3], TOKEN_TO_ID[PAD_TOKEN])
        silent = encode_utterance(None)
        self.assertEqual(set(silent), {TOKEN_TO_ID[PAD_TOKEN]})


class SituationTest(unittest.TestCase):
    def test_enumeration_is_stable_and_keyed(self):
        situations = enumerate_situations()
        keys = [situation.key() for situation in situations]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreater(len(keys), 40)
        for key in keys:
            self.assertEqual(Situation.from_key(key).key(), key)

    def test_all_templates_valid_and_satisfy_requirements(self):
        for situation in enumerate_situations():
            variants = template_variants(situation)
            self.assertGreaterEqual(len(variants), 3)
            required_all, required_any = required_tokens(situation)
            for words in variants:
                validate_utterance(" ".join(words))
                self.assertTrue(required_all <= set(words))
                if required_any:
                    self.assertTrue(required_any & set(words))

    def test_label_reveals_world_assigned_kind(self):
        poison_berry = Situation("label", (("surface", "berry"), ("kind", "danger")))
        required_all, _ = required_tokens(poison_berry)
        self.assertIn("danger", required_all)
        food_berry = Situation("label", (("surface", "berry"), ("kind", "food")))
        required_all, _ = required_tokens(food_berry)
        self.assertIn("food", required_all)

    def test_each_consumable_surface_supports_every_bodily_label(self):
        for surface in ("water", "spring", "berry", "roots", "mushroom"):
            self.assertEqual(
                set(SURFACE_KINDS[surface]), {"food", "water", "danger"}
            )


class GeneratorPiecesTest(unittest.TestCase):
    def test_parse_variant_texts_json_and_lines(self):
        self.assertEqual(
            parse_variant_texts(' ["water here", "go north"] '),
            ["water here", "go north"],
        )
        self.assertEqual(
            parse_variant_texts('"water here",\n"go north"'),
            ["water here", "go north"],
        )

    def test_filter_valid_enforces_contract(self):
        situation = Situation("label", (("surface", "berry"), ("kind", "food")))
        valid, rejected = filter_valid(
            situation,
            ["berry food", "berry banana", "this berry", "berry food"],
        )
        self.assertEqual(valid, [("berry", "food")])
        self.assertEqual(rejected, 2)

    def test_build_prompt_mentions_requirements(self):
        situation = Situation("label", (("surface", "berry"), ("kind", "food")))
        prompt = build_prompt(situation, 8)
        self.assertIn("food", prompt)
        self.assertIn("JSON array", prompt)


class BankTest(unittest.TestCase):
    def _template_bank(self) -> UtteranceBank:
        return UtteranceBank(
            {
                situation.key(): template_variants(situation)
                for situation in enumerate_situations()
            }
        )

    def test_template_bank_validates_and_roundtrips(self):
        bank = self._template_bank()
        bank.validate()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            bank.save(path)
            loaded = UtteranceBank.load(path)
            self.assertEqual(set(loaded.keys), set(bank.keys))

    def test_sampling_is_seed_deterministic(self):
        bank = self._template_bank()
        situation = enumerate_situations()[0]
        first = [bank.sample(situation, random.Random(7)) for _ in range(5)]
        second = [bank.sample(situation, random.Random(7)) for _ in range(5)]
        self.assertEqual(first, second)

    def test_default_bank_covers_all_consumable_surface_labels(self):
        bank = load_default_bank()
        for surface in ("water", "spring", "berry", "roots", "mushroom"):
            for kind in ("food", "water", "danger"):
                situation = Situation(
                    "label", (("surface", surface), ("kind", kind))
                )
                variants = bank.variants(situation)
                self.assertGreaterEqual(len(variants), 3)
                self.assertTrue(all(kind in words for words in variants))

    def test_same_act_sampling_is_seeded_and_ignores_grounded_slots(self):
        bank = self._template_bank()
        food = Situation("label", (("surface", "berry"), ("kind", "food")))
        danger = Situation(
            "label", (("surface", "mushroom"), ("kind", "danger"))
        )
        first_rng = random.Random(37)
        second_rng = random.Random(37)
        first = [bank.sample_from_act(food.act, first_rng) for _ in range(50)]
        second = [bank.sample_from_act(danger.act, second_rng) for _ in range(50)]
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 8)

        label_variants = {
            words
            for situation in enumerate_situations()
            if situation.act == "label"
            for words in bank.variants(situation)
        }
        self.assertTrue(all(words in label_variants for words in first))


if __name__ == "__main__":
    unittest.main()
