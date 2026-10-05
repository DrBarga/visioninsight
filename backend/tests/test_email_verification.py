import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from test_processor_e2e import make_video


class EmailVerificationTests(unittest.TestCase):
    def test_production_account_must_verify_before_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = replace(
                Settings.from_env(), environment="production", data_dir=root / "data",
                runs_dir=root / "runs",
                database_url="sqlite:///" + (root / "data" / "test.sqlite3").as_posix(),
                embedded_worker=False, cookie_secure=False, public_base_url="http://testserver",
                smtp_host="mail.example.com", smtp_user="user", smtp_password="password",
                smtp_from="VisionInsight <no-reply@example.com>",
            )
            video = root / "video.mp4"
            make_video(video)
            app = create_app(settings)
            with patch("app.api.v1.auth.send_verification") as send, TestClient(app) as client:
                registered = client.post("/v1/auth/register", json={
                    "email": "verify@example.com", "password": "strong-password-123",
                })
                self.assertEqual(registered.status_code, 202, registered.text)
                self.assertNotIn("csrf_token", registered.json())
                duplicate = client.post("/v1/auth/register", json={
                    "email": "verify@example.com", "password": "strong-password-123",
                })
                self.assertEqual(duplicate.status_code, registered.status_code)
                self.assertEqual(duplicate.json(), registered.json())
                signed_in = client.post("/v1/auth/login", json={
                    "email": "verify@example.com", "password": "strong-password-123",
                })
                self.assertEqual(signed_in.status_code, 200)
                blocked = client.post("/v1/analyses", headers={
                    "x-csrf-token": signed_in.json()["csrf_token"], "content-type": "video/mp4"},
                    content=video.read_bytes())
                self.assertEqual(blocked.status_code, 403)
                token = send.call_args.args[2]
                self.assertTrue(client.post("/v1/auth/verify-email", json={"token": token}).json()["verified"])
                self.assertEqual(client.post("/v1/auth/verify-email", json={"token": token}).status_code, 422)
                self.assertTrue(client.get("/v1/auth/me").json()["email_verified"])
                with patch("app.api.v1.auth.send_password_reset") as reset_email:
                    requested = client.post("/v1/auth/request-password-reset", json={"email": "verify@example.com"})
                    self.assertEqual(requested.status_code, 200)
                    reset_token = reset_email.call_args.args[2]
                changed = client.post("/v1/auth/reset-password", json={
                    "token": reset_token, "password": "a-different-strong-password",
                })
                self.assertEqual(changed.status_code, 200)
                self.assertEqual(client.post("/v1/auth/reset-password", json={
                    "token": reset_token, "password": "a-different-strong-password",
                }).status_code, 422)
                self.assertEqual(client.get("/v1/auth/me").status_code, 401)
                self.assertEqual(client.post("/v1/auth/login", json={
                    "email": "verify@example.com", "password": "a-different-strong-password",
                }).status_code, 200)


if __name__ == "__main__":
    unittest.main()
