# PL-BERT Memory Palace

Frozen pretrained ALBERT model producing 768-dim contextual phoneme embeddings from IPA character sequences. Used as a drop-in replacement for VocaloFlow's learned `PhonemeEmbedding(2820, 64)` when `config.use_plbert=True`.

## Overview

PL-BERT was pretrained on IPA phoneme sequences using a masked language modeling objective. The checkpoint (`checkpoints/step_1000000.t7`) contains an ALBERT model with `vocab_size=178`, `hidden_size=768`, 12 attention heads, 6 hidden groups. The model is always frozen during VocaloFlow training — only the 768->64 projection in `VocaloFlow/model/vocaloflow.py` is learned.

**Integration flow**: ARPAbet phonemes (from phone_set.json) -> IPA characters -> PL-BERT token IDs -> ALBERT forward pass -> 768-dim contextual embeddings -> projected to 64-dim in VocaloFlow.

## text_utils.py

Exact replica of PL-BERT's 178-token vocabulary.

### Vocabulary structure
- Index 0: `$` (padding)
- Indices 1-15: punctuation (`;:,.!?` + special chars)
- Index 16: space (word boundary)
- Indices 17-68: ASCII letters A-Z, a-z
- Indices 69-177: IPA characters (110 unique codepoints)

### `tokenize_ipa(ipa_string: str) -> list[int]`
Converts an IPA string to token IDs. Unknown characters and any char with index >= 178 map to `UNK_IDX` (37, the letter 'U').

### Known quirk — duplicate U+02BB
The original PL-BERT `_letters_ipa` string contains U+02BB (ʻ) twice. This makes `list(symbols)` have 179 entries, but `char_to_idx` has 178 unique keys (last occurrence wins, mapping ʻ to index 177). This pushes ᵻ (U+1D7B) to index 178 — out of bounds for the 178-row embedding. We replicate this bug exactly; `tokenize_ipa` clamps OOB indices to UNK.

### Constants
- `VOCAB_SIZE = 178`
- `PAD_IDX = 0`
- `SPACE_IDX = 16`
- `UNK_IDX = 37`
- All IPA characters defined by Unicode codepoints in `_letters_ipa_codepoints` for reproducibility.

## plbert.py

### `PLBertFeatureExtractor(config_path, checkpoint_path, device)`
`nn.Module` wrapper around frozen `AlbertModel`.

**Constructor**: Loads `configs/config.yml` -> `AlbertConfig` -> `AlbertModel`. Loads checkpoint, strips `module.` and `encoder.` prefixes from state dict keys. Uses `strict=False` due to `embeddings.position_ids` buffer in newer transformers. Freezes all parameters.

**Default paths**: `PL-BERT/configs/config.yml` and `PL-BERT/checkpoints/step_1000000.t7` (resolved relative to the module file).

### `extract(token_ids: list[int]) -> Tensor`
Runs ALBERT forward pass on a single sequence. Returns `(seq_len, 768)` contextual embeddings. Called with output of `text_utils.tokenize_ipa()`.

## arpabet_ipa.py

Maps VocaloFlow's `phone_set.json` entries to IPA for PL-BERT tokenization.

### `ARPABET_TO_IPA: dict[str, str]`
39-entry mapping from ARPAbet base symbols to IPA. Examples: `AA` -> `ɑ`, `AY` -> `aɪ`, `CH` -> `ʧ`, `ER` -> `ɚ`. Multi-character IPA (diphthongs/affricates like `aɪ`, `ʧ`) produce multiple PL-BERT tokens.

### `STRUCTURAL_TOKENS: set[str]`
Tokens that get zero vectors: `<PAD>`, `<SP>`, `<AP>`, `<UNK>`, `<BOW>`, `<EOW>`, `<BOS>`, `<EOS>`, `<MASK>`, `<SEP>`.

### `phone_set_entry_to_arpabet(entry: str) -> str | None`
Converts `en_AA0` -> `AA` (strips `en_` prefix and stress digit). Returns `None` for structural tokens and non-English phonemes.

### `arpabet_to_ipa(arpabet: str) -> str | None`
Looks up ARPAbet symbol in `ARPABET_TO_IPA`. Returns `None` if not found.

### `phone_set_entry_to_ipa(entry: str) -> str | None`
Convenience: `phone_set_entry_to_arpabet` then `arpabet_to_ipa`.

## precompute_features.py

Batch precomputation script that creates `plbert_features.npy` for each training chunk.

### `load_plbert_modules() -> (plbert_mod, text_utils, arpabet_ipa)`
Uses `importlib.util` to import sibling modules (directory name `PL-BERT` has a hyphen, can't use normal import).

### `precompute_chunk(chunk_dir, phone_set, extractor, text_utils, arpabet_ipa, hidden_dim=768) -> bool`
For one chunk:
1. Loads `phoneme_ids.npy` (P,)
2. Identifies English phonemes, converts to IPA characters, tracking which IPA chars map to which phoneme index
3. Concatenates all IPA chars into one string, tokenizes, runs PL-BERT once on the full sequence
4. Average-pools multi-char IPA embeddings back to per-phoneme vectors (e.g., `aɪ` -> mean of 2 embeddings)
5. Structural tokens and non-English phonemes get zero vectors
6. Saves `plbert_features.npy` (P, 768) as float16

### `main()`
CLI entry point. Iterates manifest, calls `precompute_chunk` for each row. Skips chunks that already have `plbert_features.npy` (use `--overwrite` to recompute).

**Usage**: `python PL-BERT/precompute_features.py --data-dir Data/Rachie --manifest Data/Rachie/manifest.csv`

**CLI args**: `--data-dir` (required), `--manifest` (required), `--phoneset` (default: SoulX-Singer's phone_set.json), `--device` (default: cuda if available), `--overwrite` (flag).

## inspect_checkpoint.py

Diagnostic script. Loads the `.t7` checkpoint and `configs/config.yml`, prints state dict keys, config, and model structure. Run first to verify assumptions about the checkpoint format.

## Directory Structure

```
PL-BERT/
├── README.md               # This file
├── __init__.py              # Empty package marker
├── plbert.py                # PLBertFeatureExtractor (frozen ALBERT)
├── text_utils.py            # 178-token IPA vocabulary and tokenizer
├── arpabet_ipa.py           # ARPAbet-to-IPA mapping for phone_set.json entries
├── precompute_features.py   # Batch precomputation for training data
├── inspect_checkpoint.py    # Checkpoint diagnostic tool
├── configs/
│   └── config.yml           # ALBERT model config (vocab_size=178, hidden=768)
└── checkpoints/
    └── step_1000000.t7      # Pretrained PL-BERT weights (~330MB)
```
