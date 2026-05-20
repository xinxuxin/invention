from __future__ import annotations

import re
from dataclasses import dataclass


COUNTRY_ALIASES: dict[str, str] = {
    "japan": "JP",
    "japanese": "JP",
    "japanese jurisdiction": "JP",
    "jp": "JP",
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "us": "US",
    "america": "US",
    "china": "CN",
    "cn": "CN",
    "canada": "CA",
    "ca": "CA",
    "korea": "KR",
    "south korea": "KR",
    "kr": "KR",
    "europe": "EP",
    "european": "EP",
    "ep": "EP",
    "wo": "WO",
    "wipo": "WO",
}


@dataclass(frozen=True)
class CountryFilterIntent:
    field: str
    operator: str
    keep_value: str
    delete_inverse: bool
    requires_confirmation: bool = True


def normalize_country_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = re.sub(r"[\s_-]+", " ", text.lower()).strip()
    if lowered in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[lowered]
    compact = re.sub(r"[^a-z0-9]", "", lowered)
    if compact in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[compact]
    if len(text) == 2 and text.isalpha():
        return text.upper()
    return text.upper()


def parse_country_filter_mutation(message: str) -> CountryFilterIntent | None:
    lowered = re.sub(r"[\s_-]+", " ", message.lower()).strip()
    if not lowered:
        return None
    has_mutation_verb = any(
        marker in lowered
        for marker in (
            "delete",
            "remove",
            "drop",
            "keep only",
            "filter",
            "persist",
            "keep records",
            "keep rows",
        )
    )
    if not has_mutation_verb:
        return None

    for alias, code in _alias_patterns():
        if re.search(rf"\bnon {alias}\b", lowered):
            return CountryFilterIntent(field="country", operator="eq", keep_value=code, delete_inverse=True)
        if re.search(rf"\bcountry\s*(?:!=|<>|is not|not in)\s*{alias}\b", lowered):
            return CountryFilterIntent(field="country", operator="eq", keep_value=code, delete_inverse=True)
        if re.search(rf"\bnot\s+(?:in\s+)?{alias}\b", lowered) and "country" in lowered:
            return CountryFilterIntent(field="country", operator="eq", keep_value=code, delete_inverse=True)
        if re.search(rf"\b(?:keep only|filter(?: the dataset)? to only|persist only)\s+{alias}\b", lowered):
            return CountryFilterIntent(field="country", operator="eq", keep_value=code, delete_inverse=True)
        if re.search(rf"\bonly\s+{alias}\s+(?:records|entries|rows)\b", lowered):
            return CountryFilterIntent(field="country", operator="eq", keep_value=code, delete_inverse=True)
        if re.search(rf"\bcountry\s*(?:==|=|is)\s*{alias}\b", lowered) and any(
            marker in lowered for marker in ("keep", "filter", "persist", "delete", "remove", "drop")
        ):
            return CountryFilterIntent(field="country", operator="eq", keep_value=code, delete_inverse=True)

    return None


def _alias_patterns() -> list[tuple[str, str]]:
    patterns: list[tuple[str, str]] = []
    for alias, code in sorted(COUNTRY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = re.escape(re.sub(r"[\s_-]+", " ", alias.lower()).strip())
        patterns.append((normalized, code))
    return patterns
