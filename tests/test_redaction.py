from __future__ import annotations

import unittest

from microtensor_miner_controller.redaction import REDACTED, redact_text, safe_source, sanitize


class RedactionTests(unittest.TestCase):
    def test_known_secret_and_bearer_are_redacted(self) -> None:
        text = redact_text(
            "token=abcd1234 Authorization: Bearer ey.secret.value",
            ("abcd1234",),
        )
        self.assertNotIn("abcd1234", text)
        self.assertNotIn("ey.secret.value", text)
        self.assertIn(REDACTED, text)

    def test_source_drops_query_credentials(self) -> None:
        source = safe_source("https:user:pass@example.com/model?token=abc&view=1")
        self.assertNotIn("pass", source)
        self.assertNotIn("abc", source)
        self.assertIn("view=1", source)

    def test_mapping_uses_key_redaction(self) -> None:
        result = sanitize({"WANDB_API_KEY": "secret-value", "nested": {"password": "x"}})
        self.assertEqual(result["WANDB_API_KEY"], REDACTED)
        self.assertEqual(result["nested"]["password"], REDACTED)
