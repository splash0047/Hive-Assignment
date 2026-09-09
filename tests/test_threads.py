"""Comprehensive unit tests for thread reconstruction."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from hiver_agent.data.threads import reconstruct_threads


def test_one_to_one_reply():
    """Test a basic single customer tweet with a direct brand reply."""
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    df = pd.DataFrame(
        [
            {
                "tweet_id": "1",
                "author_id": "cust_1",
                "inbound": True,
                "created_at": t0,
                "text": "Help with login please",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": "2",
                "author_id": "SpotifyCares",
                "inbound": False,
                "created_at": t0 + timedelta(minutes=5),
                "text": "Try resetting your password here.",
                "in_response_to_tweet_id": "1",
            },
        ]
    )

    pairs, stats = reconstruct_threads(df, brand_handle="SpotifyCares")
    assert len(pairs) == 1
    assert stats.threads_with_brand_reply == 1
    pair = pairs[0]
    assert pair.customer_tweet_id == "1"
    assert pair.support_tweet_id == "2"
    assert pair.customer_text == "Help with login please"
    assert pair.support_text == "Try resetting your password here."
    assert pair.brand == "SpotifyCares"
    assert pair.response_latency_seconds == 300.0


def test_multi_turn_alternating_thread():
    """Test a thread where customer and brand alternate turns."""
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    df = pd.DataFrame(
        [
            {
                "tweet_id": "10",
                "author_id": "cust_2",
                "inbound": True,
                "created_at": t0,
                "text": "App crashes on start",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": "11",
                "author_id": "SpotifyCares",
                "inbound": False,
                "created_at": t0 + timedelta(minutes=2),
                "text": "What OS version are you on?",
                "in_response_to_tweet_id": "10",
            },
            {
                "tweet_id": "12",
                "author_id": "cust_2",
                "inbound": True,
                "created_at": t0 + timedelta(minutes=4),
                "text": "iOS 16.5",
                "in_response_to_tweet_id": "11",
            },
            {
                "tweet_id": "13",
                "author_id": "SpotifyCares",
                "inbound": False,
                "created_at": t0 + timedelta(minutes=8),
                "text": "Please reinstall the app.",
                "in_response_to_tweet_id": "12",
            },
        ]
    )

    pairs, _stats = reconstruct_threads(df, brand_handle="SpotifyCares")
    assert len(pairs) == 2
    assert pairs[0].customer_tweet_id == "10"
    assert pairs[0].support_tweet_id == "11"
    assert pairs[1].customer_tweet_id == "12"
    assert pairs[1].support_tweet_id == "13"


def test_multiple_replies_to_same_tweet():
    """Test where brand sends two answers or two customers reply to same tweet."""
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    df = pd.DataFrame(
        [
            {
                "tweet_id": "20",
                "author_id": "cust_3",
                "inbound": True,
                "created_at": t0,
                "text": "Where is my refund?",
                "in_response_to_tweet_id": None,
            },
            {
                "tweet_id": "21",
                "author_id": "SpotifyCares",
                "inbound": False,
                "created_at": t0 + timedelta(minutes=1),
                "text": "First response",
                "in_response_to_tweet_id": "20",
            },
            {
                "tweet_id": "22",
                "author_id": "SpotifyCares",
                "inbound": False,
                "created_at": t0 + timedelta(minutes=3),
                "text": "Followup response",
                "in_response_to_tweet_id": "20",
            },
        ]
    )

    pairs, _stats = reconstruct_threads(df, brand_handle="SpotifyCares")
    assert len(pairs) == 1
    # Picks the earliest response
    assert pairs[0].support_tweet_id == "21"


def test_missing_parent():
    """Test orphan replies when parent tweet is missing from dataset."""
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    df = pd.DataFrame(
        [
            {
                "tweet_id": "31",
                "author_id": "SpotifyCares",
                "inbound": False,
                "created_at": t0,
                "text": "Glad that helped!",
                "in_response_to_tweet_id": "99999",  # Not in df
            },
        ]
    )

    pairs, stats = reconstruct_threads(df, brand_handle="SpotifyCares")
    # No inbound customer turn in thread -> no pair extracted
    assert len(pairs) == 0
    assert stats.missing_parent_count >= 1


def test_cycle_detection():
    """Test handling of cyclic parent-child references without hanging."""
    t0 = datetime(2023, 1, 1, 10, 0, 0)
    df = pd.DataFrame(
        [
            {
                "tweet_id": "40",
                "author_id": "cust_4",
                "inbound": True,
                "created_at": t0,
                "text": "Loop 1",
                "in_response_to_tweet_id": "41",
            },
            {
                "tweet_id": "41",
                "author_id": "SpotifyCares",
                "inbound": False,
                "created_at": t0 + timedelta(minutes=1),
                "text": "Loop 2",
                "in_response_to_tweet_id": "40",
            },
        ]
    )

    _pairs, stats = reconstruct_threads(df)
    assert stats.cycles_detected >= 1
