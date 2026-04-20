"""ARPAbet-to-IPA conversion for mapping VocaloFlow phonemes to PL-BERT tokens.

VocaloFlow's phone_set.json uses entries like 'en_AA0', 'en_B', 'en_CH'.
PL-BERT expects IPA character sequences. This module bridges the two.
"""

import re

ARPABET_TO_IPA: dict[str, str] = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "ʧ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɚ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "ʤ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

_STRESS_RE = re.compile(r"[012]$")
_EN_PREFIX_RE = re.compile(r"^en_")

STRUCTURAL_TOKENS = {"<PAD>", "<SP>", "<AP>", "<UNK>", "<BOW>", "<EOW>",
                     "<BOS>", "<EOS>", "<MASK>", "<SEP>"}


def phone_set_entry_to_arpabet(entry: str) -> str | None:
    """Convert a phone_set.json entry to its base ARPAbet symbol.

    Examples:
        'en_AA0' -> 'AA'
        'en_B'   -> 'B'
        'en_CH'  -> 'CH'
        '<PAD>'  -> None  (structural token)
        'yue_aa1' -> None (non-English)

    Returns None for structural tokens and non-English phonemes.
    """
    if entry in STRUCTURAL_TOKENS:
        return None
    if not entry.startswith("en_"):
        return None
    base = _EN_PREFIX_RE.sub("", entry)
    base = _STRESS_RE.sub("", base)
    return base


def arpabet_to_ipa(arpabet: str) -> str | None:
    """Convert a base ARPAbet symbol to its IPA string.

    Returns None if the symbol is not in the mapping.
    """
    return ARPABET_TO_IPA.get(arpabet)


def phone_set_entry_to_ipa(entry: str) -> str | None:
    """Convert a phone_set.json entry directly to IPA.

    Returns None for structural/non-English tokens.
    """
    base = phone_set_entry_to_arpabet(entry)
    if base is None:
        return None
    return arpabet_to_ipa(base)
