from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as functional

    from training.train_extract import (
        Collator,
        EncodedDataset,
        bind_entity_token_weights,
        canonicalize_gold,
        parse_args,
        weighted_causal_lm_loss,
    )
except ModuleNotFoundError as exc:
    torch = None
    functional = None
    TRAINING_DEPENDENCY_ERROR = str(exc)
else:
    TRAINING_DEPENDENCY_ERROR = ""


class CharacterTokenizer:
    """Minimal fast-tokenizer fake with one token per rendered character."""

    pad_token_id = 0

    def __init__(self) -> None:
        self.complete_renderings: list[str] = []
        self.offset_requests = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        if tokenize or enable_thinking:
            raise AssertionError("unexpected chat-template mode")
        prefix = f"<user>{messages[0]['content']}</user><assistant>"
        if add_generation_prompt:
            return prefix
        rendered = prefix + messages[1]["content"] + "</assistant>"
        self.complete_renderings.append(rendered)
        return rendered

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        if add_special_tokens:
            raise AssertionError("special tokens must stay disabled")
        fields = {"input_ids": [index + 1 for index in range(len(text))]}
        if return_offsets_mapping:
            self.offset_requests += 1
            fields["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return SimpleNamespace(**fields)


def row(gold: object) -> dict[str, object]:
    return {"ref": "public-row", "prompt": "Extract: aspirin", "gold": gold}


@unittest.skipIf(
    torch is None, f"optional training dependencies unavailable: {TRAINING_DEPENDENCY_ERROR}"
)
class CanonicalizationTests(unittest.TestCase):
    def test_deduplicates_exact_pairs_with_stable_first_or_sorted_order(self) -> None:
        raw = {
            "entities": [
                {"type": "Disease", "text": "rash"},
                {"text": "aspirin", "type": "Chemical"},
                {"text": "rash", "type": "Disease"},
                {"text": "aspirin", "type": "Disease"},
            ]
        }
        first = canonicalize_gold(raw, "first")
        sorted_target = canonicalize_gold(json.dumps(raw), "sorted")

        self.assertEqual(
            first.content,
            '{"entities":[{"text":"rash","type":"Disease"},'
            '{"text":"aspirin","type":"Chemical"},'
            '{"text":"aspirin","type":"Disease"}]}',
        )
        self.assertEqual(
            sorted_target.content,
            '{"entities":[{"text":"aspirin","type":"Chemical"},'
            '{"text":"aspirin","type":"Disease"},'
            '{"text":"rash","type":"Disease"}]}',
        )
        self.assertEqual(
            [first.content[start:end] for start, end in first.entity_text_spans],
            ["rash", "aspirin", "aspirin"],
        )

    def test_entity_spans_cover_exact_json_encoding(self) -> None:
        target = canonicalize_gold(
            {"entities": [{"text": 'alpha"beta', "type": "Chemical"}]}
        )
        start, end = target.entity_text_spans[0]
        self.assertEqual(target.content[start:end], 'alpha\\"beta')
        self.assertEqual(json.loads(target.content)["entities"][0]["text"], 'alpha"beta')

    def test_never_repairs_malformed_entities(self) -> None:
        malformed = (
            "not-json",
            {"entities": "aspirin"},
            {"entities": [{"text": " aspirin", "type": "Chemical"}]},
            {"entities": [{"text": "aspirin", "type": ""}]},
            {"entities": [{"text": "aspirin", "type": "Chemical", "extra": True}]},
            {"entities": [None]},
        )
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonicalize_gold(value)


@unittest.skipIf(
    torch is None, f"optional training dependencies unavailable: {TRAINING_DEPENDENCY_ERROR}"
)
class OffsetAndDatasetTests(unittest.TestCase):
    def test_offset_binding_weights_only_entity_tokens(self) -> None:
        weights = bind_entity_token_weights(
            [(0, 2), (2, 4), (4, 6), (6, 8)],
            [-100, 10, 11, 12],
            [(2, 5)],
            4.0,
        )
        self.assertEqual(weights, [0.0, 4.0, 4.0, 1.0])

    def test_offset_binding_fails_on_gap_or_masked_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "gap"):
            bind_entity_token_weights(
                [(0, 1), (2, 3), (4, 5)],
                [-100, 1, 1],
                [(2, 5)],
                2.0,
            )
        with self.assertRaisesRegex(ValueError, "masked prompt"):
            bind_entity_token_weights([(0, 3)], [-100], [(1, 2)], 2.0)

    def test_weighted_dataset_uses_offsets_and_preserves_prompt_mask(self) -> None:
        tokenizer = CharacterTokenizer()
        dataset = EncodedDataset(
            [
                row(
                    {
                        "entities": [
                            {"text": "aspirin", "type": "Chemical"},
                            {"text": "aspirin", "type": "Chemical"},
                        ]
                    }
                )
            ],
            tokenizer,
            512,
            gold_canonicalization="first",
            entity_text_token_weight=3.0,
        )
        item = dataset[0]
        rendering = tokenizer.complete_renderings[0]
        entity_start = rendering.index('"aspirin"') + 1
        entity_end = entity_start + len("aspirin")

        self.assertEqual(tokenizer.offset_requests, 1)
        self.assertTrue(torch.all(item["labels"][: rendering.index('{')] == -100))
        self.assertTrue(torch.all(item["loss_weights"][: rendering.index('{')] == 0))
        self.assertTrue(torch.all(item["loss_weights"][entity_start:entity_end] == 3.0))
        self.assertEqual(float(item["loss_weights"][entity_start - 1]), 1.0)
        self.assertEqual(float(item["loss_weights"][entity_end]), 1.0)

    def test_default_dataset_is_byte_compatible_and_does_not_request_offsets(self) -> None:
        tokenizer = CharacterTokenizer()
        raw_gold = '{ "entities": [ { "type": "Chemical", "text": "aspirin" } ] }'
        dataset = EncodedDataset([row(raw_gold)], tokenizer, 512)
        item = dataset[0]

        self.assertIn(raw_gold, tokenizer.complete_renderings[0])
        self.assertNotIn("loss_weights", item)
        self.assertEqual(tokenizer.offset_requests, 0)
        self.assertEqual(set(item), {"input_ids", "labels"})

    def test_weighting_requires_canonicalization(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires canonicalized gold"):
            EncodedDataset(
                [row('{"entities":[]}')],
                CharacterTokenizer(),
                512,
                entity_text_token_weight=2.0,
            )


@unittest.skipIf(
    torch is None, f"optional training dependencies unavailable: {TRAINING_DEPENDENCY_ERROR}"
)
class CollatorAndLossTests(unittest.TestCase):
    def test_collator_pads_weights_with_zero_and_labels_with_ignore_index(self) -> None:
        items = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([-100, 2, 3]),
                "loss_weights": torch.tensor([0.0, 1.0, 4.0]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "labels": torch.tensor([-100, 5]),
                "loss_weights": torch.tensor([0.0, 2.0]),
            },
        ]
        batch = Collator(99)(items)
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3], [4, 5, 99]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, 3], [-100, 5, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch["loss_weights"].tolist(), [[0.0, 1.0, 4.0], [0.0, 2.0, 0.0]])

    def test_weighted_loss_applies_causal_shift_and_weighted_mean(self) -> None:
        logits = torch.tensor(
            [[[3.0, 0.0], [0.0, 2.0], [1.5, -0.5], [9.0, -9.0]]],
            requires_grad=True,
        )
        labels = torch.tensor([[-100, 0, 1, 0]])
        weights = torch.tensor([[0.0, 1.0, 4.0, 2.0]])
        actual = weighted_causal_lm_loss(logits, labels, weights)
        per_token = functional.cross_entropy(
            logits[0, :3], labels[0, 1:], reduction="none"
        )
        expected = (per_token * torch.tensor([1.0, 4.0, 2.0])).sum() / 7.0

        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(float(logits.grad[0, 3].abs().sum()), 0.0)

    def test_default_cli_controls_preserve_prior_behavior(self) -> None:
        args = parse_args(["--corpus", "public.json", "--base", "base", "--out", "out"])
        self.assertEqual(args.gold_canonicalization, "none")
        self.assertEqual(args.entity_text_token_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
