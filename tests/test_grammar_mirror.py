# (C) 2026 Gervasio Zaldivar, Yuan Tian
# SPDX-License-Identifier: GPL-3.0-only

"""Keep the repository-root grammar mirror aligned with the packaged source."""

from pathlib import Path


def test_root_grammar_matches_canonical_packaged_source():
    root = Path(__file__).resolve().parents[1]
    canonical = root / "src" / "g2rins" / "data" / "g2rins.lark"
    mirror = root / "g2rins.lark"

    assert mirror.read_bytes() == canonical.read_bytes(), (
        "Grammar mirror differs from the canonical packaged source. "
        "Edit src/g2rins/data/g2rins.lark, then update the root g2rins.lark "
        "mirror to match byte-for-byte."
    )