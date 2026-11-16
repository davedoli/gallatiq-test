from __future__ import annotations

import re

ALIASES = {
    "widgeta": "WidgetA",
    "widgetb": "WidgetB",
    "widgetc": "WidgetC",
    "gadgetx": "GadgetX",
    "fakeitem": "FakeItem",
    "supergizmo": "SuperGizmo",
    "megasprocket": "MegaSprocket",
}


def canonical_item(name: str) -> str:
    """Fold printed labels onto catalog ids. 'Widget A' and 'WidgetA (rush order)' both become WidgetA."""
    base = re.sub(r"\s*\([^)]*\)", "", name or "").strip()
    key = re.sub(r"[^a-z0-9]", "", base.lower())
    return ALIASES.get(key, base)
