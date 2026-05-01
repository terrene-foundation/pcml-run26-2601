# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Sync wrappers for async Kailash engines — used in M1 before async is taught."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import polars as pl

from kailash_ml import DataExplorer
from kailash_ml import AlertConfig

if TYPE_CHECKING:
    from kailash_ml import ComparisonResult, DataProfile


def run_profile(df: pl.DataFrame, name: str = "profile") -> DataProfile:
    """Run DataExplorer.profile() synchronously.

    Wraps the async DataExplorer engine so M1 students can profile
    a DataFrame without understanding async/await.

    Args:
        df: Polars DataFrame to profile.
        name: Display name for the profile (appears in reports).

    Returns:
        DataProfile result with statistics, distributions, and quality metrics.
    """
    explorer = DataExplorer()
    return asyncio.run(explorer.profile(df, name=name))


def run_compare(
    profiles: list[DataProfile],
    names: list[str],
) -> ComparisonResult:
    """Run DataExplorer.compare() synchronously.

    Compare multiple DataProfile objects side-by-side — useful for
    before/after cleaning comparisons or cross-dataset analysis.

    Args:
        profiles: List of DataProfile objects to compare.
        names: Display names for each profile (must match length of profiles).

    Returns:
        ComparisonResult with column-level differences and summary statistics.
    """
    explorer = DataExplorer()
    return asyncio.run(explorer.compare(profiles, names=names))


def run_alerts(
    profile: DataProfile,
    alerts: list[AlertConfig],
) -> list[dict]:
    """Run DataExplorer.check_alerts() synchronously.

    Check a profiled dataset against alert thresholds (e.g., null rate > 5%,
    outlier count > 10). Returns triggered alerts for classroom discussion.

    Args:
        profile: A DataProfile from run_profile().
        alerts: List of AlertConfig thresholds to check against.

    Returns:
        List of alert dictionaries with column, metric, threshold, and actual value.
    """
    explorer = DataExplorer()
    return asyncio.run(explorer.check_alerts(profile, alerts=alerts))
