import unittest

from src.phase5_rollout import generate_hybrid_response
from src.privacy import contains_pii, redact_pii


class PrivacyTests(unittest.TestCase):
    def test_detects_and_redacts_common_identifiers(self):
        text = "Contact jane@example.com or call 987-654-3210."

        self.assertTrue(contains_pii(text))
        redacted = redact_pii(text)
        self.assertNotIn("jane@example.com", redacted)
        self.assertNotIn("987-654-3210", redacted)

    def test_forces_unlisted_phone_number_local(self):
        chunks = ["Call 987-654-3210.", "Explain supervised learning."]
        calls = {"local": [], "remote": []}

        def local_model(text):
            calls["local"].append(text)
            return "local"

        def remote_model(text):
            calls["remote"].append(text)
            return "remote"

        _, remote_chunks, actions, forced = generate_hybrid_response(
            chunks,
            [1, 1],
            [],
            local_model,
            remote_model,
        )

        self.assertEqual(actions, [0, 1])
        self.assertEqual(forced, [True, False])
        self.assertEqual(remote_chunks, [chunks[1]])
        self.assertEqual(calls["local"], [chunks[0]])
        self.assertEqual(calls["remote"], [chunks[1]])


if __name__ == "__main__":
    unittest.main()