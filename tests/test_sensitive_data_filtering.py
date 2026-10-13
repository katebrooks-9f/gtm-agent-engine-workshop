import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import data_service, gtm_agent

os.environ["LANGSMITH_TRACING"] = "false"


class SensitiveDataFilteringTests(unittest.TestCase):
    def test_get_prospect_returns_only_contact_allow_list(self):
        record = {
            "name": "Casey Example",
            "email": "casey@example.com",
            "disqualified": False,
            "billing_qualification": {"test_marker": "sentinel"},
            "annual_revenue": 1000000,
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = gtm_agent.get_prospect.func("LEAD-TEST")

        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": "LEAD-TEST",
                "name": "Casey Example",
                "email": "casey@example.com",
                "disqualified": False,
            },
        )
        serialized = json.dumps(result)
        self.assertNotIn("billing_qualification", serialized)
        self.assertNotIn("sentinel", serialized)

    def test_new_profile_excludes_billing_and_keeps_scoring_fields(self):
        record = {
            "name": "Casey Example",
            "email": "casey@example.com",
            "billing_qualification": {"test_marker": "sentinel"},
            "annual_revenue": 1000000,
        }
        with (
            patch.object(
                data_service,
                "get_profile_from_db",
                return_value={"prospect_profile": None},
            ),
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=[]),
            patch.object(data_service, "fetch_account_details", return_value=[]),
            patch.object(data_service, "fetch_tech_stack", return_value=["Snowflake"]),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            result = gtm_agent.build_prospect_profile.func("LEAD-TEST")

        profile = result["prospect_profile"]
        self.assertEqual(profile["annual_revenue"], 1000000)
        self.assertEqual(profile["tech_stack"], ["Snowflake"])
        self.assertEqual(profile["account_details"], [])
        self.assertNotIn("billing_qualification", profile)
        self.assertNotIn("sentinel", json.dumps(profile))
        save_profile.assert_called_once_with("LEAD-TEST", profile)

    def test_cached_profile_is_filtered_before_return(self):
        profile = {
            "prospect_id": "LEAD-TEST",
            "annual_revenue": 1000000,
            "tech_stack": ["Snowflake"],
            "billing_qualification": {"test_marker": "sentinel"},
        }
        with patch.dict(data_service._PROFILES, {"LEAD-TEST": profile}):
            result = data_service.get_profile_from_db("LEAD-TEST")
            cached_result = gtm_agent.build_prospect_profile.func("LEAD-TEST")

        returned_profile = result["prospect_profile"]
        self.assertEqual(returned_profile["annual_revenue"], 1000000)
        self.assertEqual(returned_profile["tech_stack"], ["Snowflake"])
        self.assertNotIn("billing_qualification", returned_profile)
        self.assertNotIn("sentinel", json.dumps(returned_profile))
        self.assertNotIn(
            "billing_qualification",
            cached_result["prospect_profile"],
        )
        self.assertNotIn("sentinel", json.dumps(cached_result))

    def test_scoring_prompt_omits_billing_from_supplied_profile(self):
        profile = {
            "annual_revenue": 1000000,
            "tech_stack": ["Snowflake"],
            "billing_qualification": {"test_marker": "sentinel"},
        }
        offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 500000,
            "description": "Example offering",
        }
        llm_result = type("Result", (), {"model_dump": lambda self: {"score": 90}})()
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = llm_result
            result = gtm_agent.score_prospect.func(profile, offering)

        self.assertEqual(result, {"score": 90})
        user_message = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertIn('"annual_revenue": 1000000', user_message)
        self.assertIn('"tech_stack": [', user_message)
        self.assertNotIn("billing_qualification", user_message)
        self.assertNotIn("sentinel", user_message)


if __name__ == "__main__":
    unittest.main()
