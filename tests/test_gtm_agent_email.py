import unittest
from types import SimpleNamespace

from gtm_agent.gtm_agent import send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={})
        self.prospect = {"name": "Taylor Prospect", "email": "taylor@example.com"}
        self.rep = {"name": "Sam Rep", "email": "sam@example.com"}

    def send(self, body):
        return send_prospect_email.func(
            self.prospect,
            "A quick follow-up",
            body,
            self.runtime,
            from_rep=self.rep,
        )

    def test_plain_text_email_sends_successfully(self):
        result = self.send("Could we schedule a pricing review next week?")

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], "Could we schedule a pricing review next week?")

    def test_attachment_claim_is_rejected(self):
        result = self.send("Attached is our pricing deck for your review.")

        self.assertEqual(result["status"], "failed")
        self.assertIn("Attachments are not supported", result["error"])


if __name__ == "__main__":
    unittest.main()
