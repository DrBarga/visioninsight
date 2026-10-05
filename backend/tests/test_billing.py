import hashlib
import hmac
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api.v1.billing import verify_paddle_signature
from app.core.config import Settings
from app.main import create_app


class BillingTests(unittest.TestCase):
    def test_signature_checks_raw_body_and_time(self):
        now = int(time.time())
        body = b'{"a":1}'
        digest = hmac.new(b"secret", str(now).encode() + b":" + body, hashlib.sha256).hexdigest()
        header = f"ts={now};h1={digest}"
        self.assertTrue(verify_paddle_signature(body, header, "secret", now))
        self.assertFalse(verify_paddle_signature(b'{"a": 1}', header, "secret", now))
        self.assertFalse(verify_paddle_signature(body, header, "secret", now + 6))

    def test_verified_subscription_controls_entitlement_and_event_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = replace(
                Settings.from_env(), data_dir=root / "data", runs_dir=root / "runs",
                database_url="sqlite:///" + (root / "data" / "test.sqlite3").as_posix(),
                embedded_worker=False, paddle_webhook_secret="test-webhook-secret",
                paddle_developer_price_id="pri_developer", paddle_team_price_id="pri_team",
                paddle_developer_display_price="EUR 19 / month", paddle_team_display_price="EUR 49 / month",
                paddle_api_key="sandbox-key", legal_terms_url="https://example.com/terms",
                legal_privacy_url="https://example.com/privacy", business_contact_email="owner@example.com",
            )
            app = create_app(settings)
            with TestClient(app) as client:
                registered = client.post("/v1/auth/register", json={
                    "email": "billing@example.com", "password": "strong-password-123",
                }).json()
                user_id = registered["id"]
                with patch("app.api.v1.billing._request_paddle", return_value={
                    "id": "txn_test_123", "checkout": {"url": "https://checkout.example.com/pay"},
                }) as create_transaction:
                    checkout = client.post("/v1/billing/checkout", json={"plan": "developer"},
                                           headers={"x-csrf-token": registered["csrf_token"]})
                self.assertEqual(checkout.status_code, 200, checkout.text)
                self.assertEqual(checkout.json()["checkout_url"], "https://checkout.example.com/pay")
                self.assertEqual(create_transaction.call_args.args[3]["custom_data"]["visioninsight_user_id"], user_id)
                self.assertEqual(client.get("/v1/auth/me").json()["plan"], "free")

                def deliver(event_id, status, occurred_at, transaction_id="txn_test_123"):
                    event = {
                        "event_id": event_id, "event_type": "subscription.updated",
                        "occurred_at": occurred_at,
                        "data": {"id": "sub_123", "transaction_id": transaction_id,
                                 "customer_id": "ctm_123", "status": status,
                                 "custom_data": {"visioninsight_user_id": user_id},
                                 "items": [{"price": {"id": "pri_developer"}}]},
                    }
                    body = json.dumps(event, separators=(",", ":")).encode()
                    now = str(int(time.time()))
                    digest = hmac.new(settings.paddle_webhook_secret.encode(),
                                      now.encode() + b":" + body, hashlib.sha256).hexdigest()
                    return client.post("/v1/billing/webhooks/paddle", content=body,
                                       headers={"Paddle-Signature": f"ts={now};h1={digest}"})

                newer = "2026-09-29T12:00:00Z"
                older = "2026-09-29T11:00:00Z"
                self.assertEqual(deliver("evt_unknown", "active", newer, "txn_other").status_code, 422)
                self.assertEqual(deliver("evt_1", "active", newer).status_code, 200)
                self.assertEqual(client.get("/v1/auth/me").json()["plan"], "developer")
                key_response = client.post("/v1/account/api-keys", json={"name": "integration"},
                                           headers={"x-csrf-token": registered["csrf_token"]})
                self.assertEqual(key_response.status_code, 201, key_response.text)
                api_headers = {"authorization": "Bearer " + key_response.json()["key"]}
                self.assertEqual(client.get("/v1/analyses", headers=api_headers).status_code, 200)
                self.assertEqual(deliver("evt_1", "canceled", newer).status_code, 200)
                self.assertEqual(deliver("evt_2", "canceled", older).status_code, 200)
                self.assertEqual(client.get("/v1/auth/me").json()["plan"], "developer")
                self.assertEqual(deliver("evt_3", "canceled", "2026-09-29T13:00:00Z").status_code, 200)
                self.assertEqual(client.get("/v1/auth/me").json()["plan"], "free")
                self.assertEqual(client.get("/v1/analyses", headers=api_headers).status_code, 403)
                self.assertEqual(client.post("/v1/billing/webhooks/paddle", content=b"{}",
                                             headers={"Paddle-Signature": "bad"}).status_code, 401)


if __name__ == "__main__":
    unittest.main()
