from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage

from app.core.config import Settings


def email_ready(settings: Settings) -> bool:
    return bool(settings.smtp_host and settings.smtp_user and
                settings.smtp_password and settings.smtp_from)


def _send(settings: Settings, recipient: str, subject: str, body: str) -> None:
    if not email_ready(settings):
        raise RuntimeError("Email delivery is not configured")
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.smtp_from
    message["To"] = recipient
    message.set_content(body)
    context = ssl.create_default_context()
    if settings.smtp_port == 465:
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=15, context=context) as smtp:
            smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
            smtp.starttls(context=context)
            smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(message)


def send_verification(settings: Settings, recipient: str, token: str) -> None:
    _send(settings, recipient, "Verify your VisionInsight email",
        "Confirm this email address to use VisionInsight:\n\n"
        f"{settings.public_base_url}/verify?token={token}\n\n"
        "This link expires in 24 hours. If you did not create an account, ignore this message."
    )


def send_password_reset(settings: Settings, recipient: str, token: str) -> None:
    _send(settings, recipient, "Reset your VisionInsight password",
          "Use this link to set a new password:\n\n"
          f"{settings.public_base_url}/reset?token={token}\n\n"
          "This link expires in one hour. If you did not request it, ignore this message."
    )
