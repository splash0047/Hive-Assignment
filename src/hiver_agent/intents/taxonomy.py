"""Intent taxonomy schema, definitions, and loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hiver_agent.config import resolve_path


@dataclass
class IntentDefinition:
    """Definition of a single customer support intent."""

    label: str
    description: str
    positive_signals: list[str] = field(default_factory=list)
    negative_signals: list[str] = field(default_factory=list)
    example_phrases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "description": self.description,
            "positive_signals": self.positive_signals,
            "negative_signals": self.negative_signals,
            "example_phrases": self.example_phrases,
        }


@dataclass
class IntentTaxonomy:
    """Versioned taxonomy of customer support intents."""

    version: str
    brand: str
    intents: list[IntentDefinition] = field(default_factory=list)

    def labels(self) -> list[str]:
        return [i.label for i in self.intents]

    def get(self, label: str) -> IntentDefinition | None:
        for i in self.intents:
            if i.label.lower() == label.lower():
                return i
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "brand": self.brand,
            "intents": [i.to_dict() for i in self.intents],
        }

    def save_json(self, output_path: str | Path) -> Path:
        path = resolve_path(str(output_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load_json(cls, input_path: str | Path) -> IntentTaxonomy:
        path = resolve_path(str(input_path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        intents = [IntentDefinition(**item) for item in data.get("intents", [])]
        return cls(version=data.get("version", "v1"), brand=data.get("brand", ""), intents=intents)


def get_default_spotify_taxonomy() -> IntentTaxonomy:
    """Pre-built, domain-validated taxonomy for SpotifyCares."""
    intents = [
        IntentDefinition(
            label="login_account_access",
            description="Trouble logging in, password reset, 2FA, email verification, or locked account.",
            positive_signals=[
                "password",
                "login",
                "log in",
                "reset",
                "can't sign in",
                "verification code",
                "locked out",
            ],
            negative_signals=["billing", "charged", "student discount"],
            example_phrases=[
                "I forgot my password and email is not receiving reset link",
                "Locked out of my account",
            ],
        ),
        IntentDefinition(
            label="subscription_billing",
            description="Charges, unexpected billing, payment failed, receipt, currency, or price changes.",
            positive_signals=[
                "charged",
                "billing",
                "credit card",
                "bank",
                "receipt",
                "deducted",
                "double charged",
            ],
            negative_signals=["playback error", "song missing"],
            example_phrases=[
                "Why was I charged twice this month?",
                "My card was debited but account shows Free",
            ],
        ),
        IntentDefinition(
            label="plan_discount_management",
            description="Family plan invites, Duo plan, Student verification (SheerID), upgrading or downgrading plans.",
            positive_signals=[
                "family plan",
                "duo",
                "student",
                "sheerid",
                "upgrade",
                "invite member",
                "address match",
            ],
            negative_signals=["cancellation"],
            example_phrases=[
                "How do I add my partner to Duo?",
                "My student discount verification failed on SheerID",
            ],
        ),
        IntentDefinition(
            label="playback_technical_issue",
            description="Songs pausing, crashing, offline downloads failing, local files, audio skipping, or error codes.",
            positive_signals=[
                "crashes",
                "skipping",
                "pauses",
                "won't play",
                "download error",
                "offline not working",
                "blank screen",
            ],
            negative_signals=["account hack", "subscription price"],
            example_phrases=[
                "App keeps crashing on startup on Android 14",
                "Songs stop playing after 10 seconds",
            ],
        ),
        IntentDefinition(
            label="device_connectivity",
            description="Bluetooth, Spotify Connect, Alexa, smart speaker, CarPlay, PlayStation, or TV app issues.",
            positive_signals=[
                "connect",
                "bluetooth",
                "carplay",
                "alexa",
                "echo",
                "ps5",
                "tv",
                "speaker",
                "chromecast",
            ],
            negative_signals=["refund"],
            example_phrases=[
                "Spotify Connect cannot discover my Sonos speakers",
                "CarPlay disconnects randomly",
            ],
        ),
        IntentDefinition(
            label="content_playlist_availability",
            description="Missing songs, greyed out tracks, lyrics not showing, explicit content toggle, or playlist loss.",
            positive_signals=[
                "greyed out",
                "song missing",
                "album not available",
                "lyrics",
                "lost playlist",
                "explicit filter",
            ],
            negative_signals=["credit card"],
            example_phrases=[
                "Why is this album greyed out in my country?",
                "My playlist disappeared overnight",
            ],
        ),
        IntentDefinition(
            label="cancellation_refund",
            description="Cancelling Premium subscription, switching to Free, refund requests.",
            positive_signals=[
                "cancel",
                "cancel subscription",
                "refund",
                "stop renewal",
                "money back",
            ],
            negative_signals=["app crash"],
            example_phrases=[
                "I want to cancel my premium and get a refund",
                "Please cancel my auto-renew",
            ],
        ),
        IntentDefinition(
            label="security_compromised_account",
            description="Unauthorized changes, stranger playing music on account, email changed without permission, hacked.",
            positive_signals=[
                "hacked",
                "compromised",
                "someone else listening",
                "email changed",
                "unauthorized login",
            ],
            negative_signals=["normal password reset"],
            example_phrases=[
                "Someone in another country is using my account and changed my email",
                "My account got hacked",
            ],
        ),
        IntentDefinition(
            label="how_to_feature_request",
            description="Inquiries about how features work, UI navigation, feature suggestions, or feedback.",
            positive_signals=[
                "how do i",
                "where is the button",
                "feature request",
                "bring back",
                "suggestion",
            ],
            negative_signals=["urgent outage"],
            example_phrases=[
                "How do I enable crossfade on desktop?",
                "Can you add folder support for mobile playlists?",
            ],
        ),
        IntentDefinition(
            label="service_outage",
            description="Widespread outage reports, server errors (500s), service down for everyone.",
            positive_signals=[
                "is spotify down",
                "server error",
                "down for everyone",
                "service unavailable",
                "outage",
            ],
            negative_signals=["single device Bluetooth issue"],
            example_phrases=[
                "Is the service down right now? Nothing loads for anyone",
                "Getting error 500 across web player",
            ],
        ),
        IntentDefinition(
            label="other_unclear",
            description="Ambiguous, spam, non-English, greetings without question, or out-of-scope messages.",
            positive_signals=["hello", "hi", "hey", "test", "anyone there"],
            negative_signals=[],
            example_phrases=["Hello?", "Yo check out my music @SpotifyCares"],
        ),
    ]
    return IntentTaxonomy(version="v1", brand="SpotifyCares", intents=intents)
