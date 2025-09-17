from typing import Tuple

UNSAFE_KEYWORDS = [
    "disable safety", "bypass emissions", "delete egr", "defeat o2",
    "illegal", "street racing setup", "tamper"
]

def is_safe(text: str) -> Tuple[bool, str]:
    lowered = text.lower()
    for kw in UNSAFE_KEYWORDS:
        if kw in lowered:
            return False, f"Blocked by safety policy (keyword: {kw})"
    return True, ""
