from __future__ import annotations

import json
import os
import random
import stat
import tempfile
import unittest
from pathlib import Path

from training.prepare_imatrix import (
    GoldValidationError,
    TrainingAPI,
    canonical_gold,
    prepare_calibration,
)

TEMPLATE = (
    "{% set enable_thinking = enable_thinking %}"
    "<|im_start|>{{ role }}<|im_end|>"
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.chat_template = TEMPLATE
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return (
            f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n{messages[1]['content']}<|im_end|>\n"
        )


def make_row(index: int) -> dict[str, object]:
    entity = f"Entity{index}"
    return {
        "ref": f"row-{index:05d}",
        "partition": "train",
        "prompt": f"Extract entities. Text: {entity}",
        "inputs": {"text": entity},
        "gold": json.dumps({"entities": [{"type": "Chemical", "text": entity}]}),
    }


class PrepareImatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "public.json"
        self.corpus.write_text('{"fixture":true}\n', encoding="utf-8")
        self.tokenizer_dir = self.root / "tokenizer"
        self.tokenizer_dir.mkdir()
        (self.tokenizer_dir / "tokenizer.json").write_text('{"version":"fake"}\n')
        (self.tokenizer_dir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "Qwen2Tokenizer", "chat_template": TEMPLATE}) + "\n",
            encoding="utf-8",
        )
        (self.tokenizer_dir / "config.json").write_text(
            '{"model_type":"qwen3"}\n', encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def api(rows: list[dict[str, object]]) -> TrainingAPI:
        return TrainingAPI("sha256:fixture", "fake.load_rows", lambda _path: list(rows))

    def run_prepare(
        self,
        rows: list[dict[str, object]],
        output: Path,
        *,
        reserve: int = 0,
        maximum: int | None = None,
        tokenizer: FakeTokenizer | None = None,
    ) -> tuple[dict[str, object], FakeTokenizer]:
        fake = tokenizer or FakeTokenizer()
        metadata = prepare_calibration(
            corpus=self.corpus,
            tokenizer_path=self.tokenizer_dir,
            output=output,
            reserve_examples=reserve,
            max_examples=maximum,
            training_api=self.api(rows),
            tokenizer_loader=lambda _path: fake,
        )
        return metadata, fake

    def test_canonical_gold_deduplicates_and_sorts_exact_pairs(self) -> None:
        row = {
            "prompt": "Text: Zinc and acne",
            "inputs": {"text": "Zinc and acne"},
            "gold": {
                "entities": [
                    {"text": "acne", "type": "Disease"},
                    {"text": "Zinc", "type": "Chemical"},
                    {"text": "acne", "type": "Disease"},
                ]
            },
        }
        self.assertEqual(
            canonical_gold(row),
            '{"entities":[{"text":"Zinc","type":"Chemical"},'
            '{"text":"acne","type":"Disease"}]}',
        )

    def test_malformed_and_non_substring_gold_are_rejected(self) -> None:
        malformed = make_row(1)
        malformed["gold"] = "not-json"
        with self.assertRaisesRegex(GoldValidationError, "malformed_gold_json"):
            canonical_gold(malformed)

        absent = make_row(2)
        absent["gold"] = '{"entities":[{"text":"invented","type":"Disease"}]}'
        with self.assertRaisesRegex(GoldValidationError, "gold_text_not_substring"):
            canonical_gold(absent)

    def test_renders_exact_chat_and_writes_regular_atomic_outputs(self) -> None:
        output = self.root / "calibration.txt"
        rows = [make_row(2), make_row(1)]
        metadata, tokenizer = self.run_prepare(rows, output)

        canonical_first = canonical_gold(rows[0])
        expected = (
            f"<|im_start|>user\n{rows[0]['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{canonical_first}<|im_end|>\n"
        )
        self.assertTrue(output.read_text(encoding="utf-8").startswith(expected))
        self.assertEqual(metadata["output"]["records"], 2)
        self.assertTrue(stat.S_ISREG(output.lstat().st_mode))
        sidecar = output.with_name(output.name + ".metadata.json")
        self.assertTrue(stat.S_ISREG(sidecar.lstat().st_mode))
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o644)
        self.assertEqual(len(tokenizer.calls), 2)
        for call in tokenizer.calls:
            self.assertIs(call["tokenize"], False)
            self.assertIs(call["add_generation_prompt"], False)
            self.assertIs(call["enable_thinking"], False)

    def test_seed92_reserve_then_cap_matches_training_shuffle(self) -> None:
        rows = [make_row(index) for index in range(400)]
        expected = list(rows)
        random.Random(92).shuffle(expected)  # noqa: S311 - deterministic fixture shuffle, not cryptographic
        expected_refs = [row["ref"] for row in expected[384:387]]

        first = self.root / "first.txt"
        second = self.root / "second.txt"
        metadata, _ = self.run_prepare(rows, first, reserve=384, maximum=3)
        repeated, _ = self.run_prepare(rows, second, reserve=384, maximum=3)

        self.assertEqual(metadata["selection"]["included_refs"], expected_refs)
        self.assertEqual(
            metadata["selection"]["reserved_refs"], [row["ref"] for row in expected[:384]]
        )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(metadata["output"], repeated["output"])

    def test_invalid_rows_are_excluded_and_bound_in_metadata(self) -> None:
        rows = [make_row(0), make_row(1), make_row(2)]
        rows[1]["gold"] = '{"entities":[{"text":"guess","type":"Chemical"}]}'
        output = self.root / "filtered.txt"
        metadata, _ = self.run_prepare(rows, output)

        self.assertEqual(metadata["selection"]["included_examples"], 2)
        self.assertEqual(len(metadata["selection"]["rejected_rows"]), 1)
        rejection = metadata["selection"]["rejected_rows"][0]
        self.assertEqual(rejection["ref"], "row-00001")
        self.assertEqual(rejection["code"], "gold_text_not_substring")
        self.assertNotIn("guess", output.read_text(encoding="utf-8"))

    def test_refuses_symlink_destination_and_invalid_settings(self) -> None:
        target = self.root / "target.txt"
        target.write_text("leave intact", encoding="utf-8")
        output = self.root / "linked.txt"
        os.symlink(target, output)
        with self.assertRaisesRegex(ValueError, "regular, non-symlink"):
            self.run_prepare([make_row(0)], output)
        self.assertEqual(target.read_text(encoding="utf-8"), "leave intact")

        with self.assertRaisesRegex(ValueError, "seed must be 92"):
            prepare_calibration(
                corpus=self.corpus,
                tokenizer_path=self.tokenizer_dir,
                output=self.root / "bad.txt",
                reserve_examples=0,
                max_examples=1,
                seed=1,
                training_api=self.api([make_row(0)]),
                tokenizer_loader=lambda _path: FakeTokenizer(),
            )


if __name__ == "__main__":
    unittest.main()
