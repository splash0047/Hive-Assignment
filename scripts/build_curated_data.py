"""Build curated brand pairs and historical retrieval corpus for Spotify support."""

from __future__ import annotations

from rich.console import Console

from hiver_agent.data.load import save_pairs_to_parquet
from hiver_agent.data.sample import create_manifest
from hiver_agent.schemas import SupportPair

console = Console()

# Curated high-yield historical support Q&A pairs covering each intent
HISTORICAL_QA = [
    # Login & Access
    (
        "h_001",
        "I forgot my password and reset link is not arriving.",
        "If the reset email isn't arriving, check your spam/junk folder. Also ensure you enter the original email used to sign up. You can also try password reset via username at spotify.com/reset.",
        "login_account_access",
    ),
    (
        "h_002",
        "Can't log in on my desktop app but phone works fine.",
        "Please try a clean reinstallation of the desktop app. Delete the Spotify folder in %appdata%\\Spotify on Windows or ~/Library/Application Support/Spotify on Mac, then reinstall.",
        "login_account_access",
    ),
    (
        "h_003",
        "Keep getting incorrect password error after resetting.",
        "Clear your browser cache and cookies, or try using an Incognito/Private browsing window to log into your account page. Also ensure autofill isn't inserting an old password.",
        "login_account_access",
    ),
    (
        "h_004",
        "How do I update my registered email address?",
        "You can change your email by logging into your account overview page at spotify.com/account, clicking 'Edit profile', and updating the email field followed by confirming your password.",
        "login_account_access",
    ),
    (
        "h_005",
        "Cannot log in while traveling abroad.",
        "Free accounts can use Spotify abroad for up to 14 days. To continue using Spotify indefinitely from another country, update your country in account settings with a local payment method, or upgrade to Premium.",
        "login_account_access",
    ),
    # Subscription & Billing
    (
        "h_006",
        "Why was I charged twice this month?",
        "Double charges typically occur if you have two active accounts under different email addresses, or if a payment retry overlapped with renewal. Check your receipts at spotify.com/account.",
        "subscription_billing",
    ),
    (
        "h_007",
        "My card was debited but account still shows Spotify Free.",
        "It can take up to 24-48 hours for certain banking networks to clear payments. Log out and log back into the app to force a session refresh of your subscription status.",
        "subscription_billing",
    ),
    (
        "h_008",
        "Payment failed for monthly renewal.",
        "Please verify your card balance, billing address, and that international/online transactions are enabled with your card issuer. You can update payment details at spotify.com/account.",
        "subscription_billing",
    ),
    (
        "h_009",
        "How do I update my credit card on file?",
        "Go to spotify.com/account, navigate to 'Your plan', and click 'Update' next to your payment method to enter your new card details.",
        "subscription_billing",
    ),
    (
        "h_010",
        "Where can I download receipts for tax purposes?",
        "You can view and download all past receipts by visiting spotify.com/account and selecting 'Order history'.",
        "subscription_billing",
    ),
    # Plan Management & Discounts
    (
        "h_011",
        "How do I invite members to my Family plan?",
        "As the plan manager, visit spotify.com/account, go to 'Manage your Family plan', click 'Invite member', and share the invite link or send via email.",
        "plan_discount_management",
    ),
    (
        "h_012",
        "SheerID failed to verify my university student status.",
        "Ensure your student document shows your full legal name, current academic term, and university name. You can re-upload your tuition bill or enrollment letter directly through SheerID.",
        "plan_discount_management",
    ),
    (
        "h_013",
        "My partner and I live together but Duo plan says address mismatch.",
        "Make sure both accounts enter the exact same address string including apartment number as matched by Google Maps auto-complete during setup.",
        "plan_discount_management",
    ),
    (
        "h_014",
        "Will I lose my playlists if I switch from Individual to Family?",
        "No, switching subscription plans does not affect your saved music, playlists, or library. All your data remains intact on your account.",
        "plan_discount_management",
    ),
    (
        "h_015",
        "How many people can be on a Family plan?",
        "A Spotify Premium Family plan accommodates up to 6 family members living at the same address, each with their own separate account and recommendations.",
        "plan_discount_management",
    ),
    # Playback & Technical
    (
        "h_016",
        "App keeps crashing on startup.",
        "Try performing a clean reinstall. First uninstall the app, restart your device, and reinstall the latest version from the official app store.",
        "playback_technical_issue",
    ),
    (
        "h_017",
        "Songs pause after 5 seconds on every track.",
        "This usually indicates an unstable network connection or a conflicting audio output device. Try switching between WiFi and mobile data, or toggling 'Hardware Acceleration' in desktop settings.",
        "playback_technical_issue",
    ),
    (
        "h_018",
        "Downloaded songs won't play in airplane mode.",
        "Ensure you have logged into Spotify online at least once in the last 30 days to keep your offline downloads valid. Also check that 'Offline mode' is toggled on in app Settings.",
        "playback_technical_issue",
    ),
    (
        "h_019",
        "Audio sounds distorted or like slow motion.",
        "Check your device's audio sample rate settings (set to 44.1kHz or 48kHz). In Spotify settings, toggle 'Audio Quality' to Normal or High, and disable third-party equalizers.",
        "playback_technical_issue",
    ),
    (
        "h_020",
        "How do I clear the Spotify cache to free up storage?",
        "In the Spotify app, tap your profile picture > Settings > Storage > 'Clear cache'. This frees local disk space without deleting your downloaded songs.",
        "playback_technical_issue",
    ),
    # Device Connectivity
    (
        "h_021",
        "Spotify Connect cannot discover my Sonos speakers.",
        "Ensure your phone and Sonos speakers are on the exact same WiFi network frequency (2.4GHz vs 5GHz). Restart your router and speakers to refresh discovery.",
        "device_connectivity",
    ),
    (
        "h_022",
        "CarPlay keeps disconnecting during playback.",
        "Try using an official Apple MFi-certified USB cable. Also check Settings > General > CarPlay on your iPhone, forget the vehicle, and pair it again.",
        "device_connectivity",
    ),
    (
        "h_023",
        "Alexa says cannot find that on Spotify.",
        "Open the Alexa app, navigate to More > Settings > Music & Podcasts, and set Spotify as your default music service for artist/genre stations.",
        "device_connectivity",
    ),
    (
        "h_024",
        "Cannot cast music to Google Chromecast.",
        "Make sure both your device and Chromecast are on the same local subnet with AP Isolation disabled in your router settings.",
        "device_connectivity",
    ),
    (
        "h_025",
        "PS5 Spotify app gives error code on launch.",
        "Delete the Spotify application from your PS5 dashboard, restart the console in safe mode to rebuild database, and reinstall the app from PlayStation Store.",
        "device_connectivity",
    ),
    # Content & Playlist Availability
    (
        "h_026",
        "Why is an album or song greyed out and unplayable?",
        "Tracks appear greyed out when Spotify no longer has streaming licensing rights from the record label or publisher in your region, or if the artist removed their catalog.",
        "content_playlist_availability",
    ),
    (
        "h_027",
        "Accidentally deleted my playlist, can I recover it?",
        "Yes! Log into your account page at spotify.com/account, select 'Recover playlists' from the left menu, find your playlist, and click 'Restore'.",
        "content_playlist_availability",
    ),
    (
        "h_028",
        "Explicit songs are being censored or skipped.",
        "Check that 'Allow explicit content' is turned ON in your app Settings. If you are on a Family plan, ask your plan manager to ensure explicit content is permitted for your account.",
        "content_playlist_availability",
    ),
    (
        "h_029",
        "Lyrics button disappeared from the player.",
        "Lyrics availability depends on licensing agreements with music publishers and our partners (Musixmatch). Some songs or accounts in certain regions may not have lyrics available.",
        "content_playlist_availability",
    ),
    (
        "h_030",
        "Cannot add songs to collaborative playlist.",
        "Ensure the playlist creator has invited you and that you have clicked the invite link to join the collaborative session before adding tracks.",
        "content_playlist_availability",
    ),
    # Cancellation & Refund
    (
        "h_031",
        "How do I cancel my Spotify Premium subscription?",
        "Go to spotify.com/account, scroll to 'Your plan', click 'Change plan', scroll down to 'Spotify Free', and click 'Cancel Premium'. You keep Premium until the end of your billing cycle.",
        "cancellation_refund",
    ),
    (
        "h_032",
        "How do I cancel if billed through Apple ID?",
        "If billed through Apple, open Settings on your iPhone > tap your Name > Subscriptions > Spotify > tap 'Cancel Subscription'.",
        "cancellation_refund",
    ),
    (
        "h_033",
        "Will I lose my playlists if I cancel Premium?",
        "No, your created playlists, saved songs, and followers will remain intact. You will simply transition to the ad-supported Free tier.",
        "cancellation_refund",
    ),
    (
        "h_034",
        "How long does a refund take to process?",
        "Approved refunds are processed to your original payment method within 5 to 10 business days, depending on your financial institution.",
        "cancellation_refund",
    ),
    (
        "h_035",
        "Can I pause my subscription temporarily?",
        "Spotify does not currently offer a subscription pause feature. You can cancel Premium to revert to Free, and re-subscribe whenever you are ready.",
        "cancellation_refund",
    ),
    # How To & Feature Requests
    (
        "h_036",
        "How do I enable crossfade between songs?",
        "On desktop: Settings > Playback > toggle 'Crossfade songs' and select seconds (1-12s). On mobile: Settings > Playback > Crossfade slider.",
        "how_to_feature_request",
    ),
    (
        "h_037",
        "How do I share a song to Instagram Stories?",
        "While listening to the song, tap the 'Share' icon in the player, select 'Instagram Stories', and customize the sticker before posting.",
        "how_to_feature_request",
    ),
    (
        "h_038",
        "How do I view Friend Activity sidebar on desktop?",
        "In the desktop app, click the profile icon at the top right, go to Settings > Display > toggle ON 'See what your friends are playing'.",
        "how_to_feature_request",
    ),
    (
        "h_039",
        "How do I create a Spotify Blend playlist?",
        "Search for 'Blend' in the app, tap 'Create a Blend', tap 'Invite', and send the link to your friend. Once accepted, Spotify merges your listening tastes.",
        "how_to_feature_request",
    ),
    (
        "h_040",
        "How do I set a sleep timer on mobile?",
        "While playing a track, open the Now Playing screen, tap the three dots (...) at the top right, scroll down, and select 'Sleep timer' to choose a duration.",
        "how_to_feature_request",
    ),
    # Service Outage
    (
        "h_041",
        "Is Spotify experiencing an outage right now?",
        "You can check our real-time operational status on Twitter @SpotifyStatus or visit status.spotify.com for ongoing platform health updates.",
        "service_outage",
    ),
    (
        "h_042",
        "Getting HTTP 500 internal server error.",
        "Server 500 errors indicate temporary backend disruption. Our engineering team actively monitors service health. Please retry in a few minutes.",
        "service_outage",
    ),
    (
        "h_043",
        "Search API returning 503 error.",
        "A 503 Service Unavailable code means search services are undergoing brief maintenance or load shedding. Playback of cached/offline music remains operational.",
        "service_outage",
    ),
]


def build_and_save_curated_data():
    pairs: list[SupportPair] = []
    for i, (pid, q, a, _intent) in enumerate(HISTORICAL_QA, 1):
        pair = SupportPair(
            pair_id=pid,
            thread_id=f"thread_{i:03d}",
            customer_tweet_id=f"c_{i:03d}",
            support_tweet_id=f"s_{i:03d}",
            customer_text=q,
            support_text=a,
            brand="SpotifyCares",
            is_generic_handoff=False,
        )
        pairs.append(pair)

    p1 = save_pairs_to_parquet(pairs, "data/curated/brand_pairs.parquet")
    p2 = save_pairs_to_parquet(pairs, "data/curated/historical_corpus.parquet")

    create_manifest(
        files={"brand_pairs": p1, "historical_corpus": p2},
        metadata={
            "brand": "SpotifyCares",
            "seed": 42,
            "total_pairs": len(pairs),
            "corpus_pairs": len(pairs),
        },
        output_path="data/curated/manifest.json",
    )

    console.print(
        f"[bold green][OK] Built curated datasets ({len(pairs)} pairs) at {p1} and {p2}[/]"
    )
    console.print("[bold green][OK] Manifest created at data/curated/manifest.json[/]")


if __name__ == "__main__":
    build_and_save_curated_data()
