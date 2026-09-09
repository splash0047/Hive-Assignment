"""Unit tests for text cleaning and handoff detection."""

from __future__ import annotations

from hiver_agent.data.clean import clean_text, extract_mentions, has_urls, is_generic_handoff


def test_clean_text_mentions_and_urls():
    raw = "Hey @SpotifyCares my app is broken! See https://spotify.com/help &amp; fix it!"
    cleaned = clean_text(raw, strip_mentions=True, normalize_urls=True)
    assert "@SpotifyCares" not in cleaned
    assert "https://" not in cleaned
    assert "&amp;" not in cleaned
    assert "& fix it!" in cleaned
    assert "app is broken" in cleaned


def test_extract_mentions():
    text = "Hello @AppleSupport and @AmazonHelp please assist"
    mentions = extract_mentions(text)
    assert mentions == ["@AppleSupport", "@AmazonHelp"]


def test_has_urls():
    assert has_urls("Check out https://t.co/abc for details")
    assert has_urls("Visit www.spotify.com")
    assert not has_urls("No links here, just plain text.")


def test_is_generic_handoff():
    assert is_generic_handoff("Please send us a direct message with your account email.")
    assert is_generic_handoff("Reach out via DM so we can look into your account.")
    assert is_generic_handoff("Can you please DM us your order ID?")
    assert not is_generic_handoff(
        "You can clear your cache by going to Settings > Storage > Clear Cache."
    )
