"""PL-BERT phoneme vocabulary — exact replica of the vocab from PL-BERT's text_utils.py.

The vocabulary is 178 symbols: pad + punctuation + ASCII letters + IPA characters.
Index 0 ('$') is padding. Index 16 (' ') is the word separator.

The original PL-BERT _letters_ipa has a duplicate U+02BB which pushes the last
character to index 178 — out of bounds for vocab_size=178. We replicate this
behavior exactly: that char maps to UNK since it was never embedded during training.
"""

_pad = "$"
# ;:,.!?¡¿—…"«»\u201c\u201d<space>  (16 chars, all unique)
_punctuation = ';:,.!?\u00a1\u00bf\u2014\u2026\u0022\u00ab\u00bb\u201c\u201d '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# fmt: off
# Each IPA character listed by Unicode codepoint for reproducibility.
# This is the exact set from PL-BERT's text_utils.py including the
# duplicate U+02BB that makes len(_all_symbols) == 179.
_letters_ipa_codepoints = [
    0x0251, 0x0250, 0x0252, 0x00E6, 0x0253, 0x0299, 0x03B2, 0x0254,  # ɑɐɒæɓʙβɔ
    0x0255, 0x00E7, 0x0257, 0x0256, 0x00F0, 0x02A4, 0x0259, 0x0258,  # ɕçɗɖðʤəɘ
    0x025A, 0x025B, 0x025C, 0x025D, 0x025E, 0x025F, 0x0284, 0x0261,  # ɚɛɜɝɞɟʄɡ
    0x0260, 0x0262, 0x029B, 0x0266, 0x0267, 0x0127, 0x0265, 0x029C,  # ɠɢʛɦɧħɥʜ
    0x0268, 0x026A, 0x029D, 0x026D, 0x026C, 0x026B, 0x026E, 0x029F,  # ɨɪʝɭɬɫɮʟ
    0x0271, 0x026F, 0x0270, 0x014B, 0x0273, 0x0272, 0x0274, 0x00F8,  # ɱɯɰŋɳɲɴø
    0x0275, 0x0278, 0x03B8, 0x0153, 0x0276, 0x0298, 0x0279, 0x027A,  # ɵɸθœɶʘɹɺ
    0x027E, 0x027B, 0x0280, 0x0281, 0x027D, 0x0282, 0x0283, 0x0288,  # ɾɻʀʁɽʂʃʈ
    0x02A7, 0x0289, 0x028A, 0x028B, 0x2C71, 0x028C, 0x0263, 0x0264,  # ʧʉʊʋⱱʌɣɤ
    0x028D, 0x03C7, 0x028E, 0x028F, 0x0291, 0x0290, 0x0292, 0x0294,  # ʍχʎʏʑʐʒʔ
    0x02A1, 0x0295, 0x02A2, 0x01C0, 0x01C1, 0x01C2, 0x01C3, 0x02C8,  # ʡʕʢǀǁǂǃˈ
    0x02CC, 0x02D0, 0x02D1, 0x02BC, 0x02B4, 0x02B0, 0x02B1, 0x02B2,  # ˌːˑʼʴʰʱʲ
    0x02B7, 0x02E0, 0x02E4, 0x02DE, 0x2193, 0x2191, 0x2192, 0x2197,  # ʷˠˤ˞↓↑→↗
    0x2198, 0x02BB, 0x0329, 0x0308, 0x02BB, 0x1D7B,                  # ↘ʻ̩̈ʻᵻ
]
# fmt: on

_letters_ipa = "".join(chr(cp) for cp in _letters_ipa_codepoints)
_all_symbols = list(_pad + _punctuation + _letters + _letters_ipa)

VOCAB_SIZE = 178

# Build char_to_idx exactly as the original: iterate in order, later
# duplicates overwrite earlier ones (U+02BB gets index 177, pushing
# U+1D7B to index 178 which is out of bounds for the embedding matrix).
char_to_idx: dict[str, int] = {s: i for i, s in enumerate(_all_symbols)}
idx_to_char: dict[int, str] = {i: s for i, s in enumerate(_all_symbols)}

PAD_IDX = 0
SPACE_IDX = char_to_idx[" "]   # 16 — word boundary
UNK_IDX = char_to_idx["U"]     # 37 — fallback for unknown chars


def tokenize_ipa(ipa_string: str) -> list[int]:
    """Convert an IPA string to PL-BERT token IDs.

    Unknown characters and any character whose index >= VOCAB_SIZE
    map to UNK_IDX (37).
    """
    tokens = []
    for c in ipa_string:
        idx = char_to_idx.get(c, UNK_IDX)
        if idx >= VOCAB_SIZE:
            idx = UNK_IDX
        tokens.append(idx)
    return tokens
