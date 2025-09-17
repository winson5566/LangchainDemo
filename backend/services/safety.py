from typing import Tuple

# List of keywords that indicate potentially unsafe or illegal queries
UNSAFE_KEYWORDS = [
    "disable safety", "bypass emissions", "delete egr", "defeat o2",
    "illegal", "street racing setup", "tamper"
]

def is_safe(text: str) -> Tuple[bool, str]:
    """
    Check whether the input text is safe by scanning for known unsafe keywords.

    Parameters:
        text (str): The input text to check.

    Returns:
        Tuple[bool, str]:
            - A boolean indicating if the text is safe (True = safe, False = unsafe).
            - A message explaining the reason if blocked, or empty string if safe.
    """
    # Convert the input to lowercase to make the check case-insensitive
    lowered = text.lower()

    # Check for the presence of any unsafe keyword
    for kw in UNSAFE_KEYWORDS:
        if kw in lowered:
            # Unsafe keyword found — block the request
            return False, f"Blocked by safety policy (keyword: {kw})"

    # No unsafe content detected
    return True, ""
