import os
from collections import Counter
from sys import prefix
from typing import List, Union, Iterable

# Class for tokenizer. Mind many of these comments are for my own tracking purposes.
class CharTokenizer:
    def __init__(self):  # Initialize the tokenizer with empty vocabularies to standardize creation.
        self.stoi = {}
        self.itos = {}
        self.pad_id = None  # pad_id and unk_id will be set when building the vocab. This is to prevent using unfitted vocabs.
        self.unk_id = None
        self.vocab_size = 0

    def is_fitted(self) -> bool:  # Quick check to see if the tokenizer is fitted.
        return bool(self.stoi) and bool(self.itos)

    def fit(self, texts_or_paths, include_specials: bool = True, min_freq: int = 1) -> None: # Build vocab from text or files.
        if self.is_fitted():  # Simple checks to prevent refitting.
            raise ValueError("Tokenizer is already fitted. Please create a new instance to fit again.")
        # Checks for file path, raw text, or list of file paths.
        if isinstance(texts_or_paths, str) and os.path.isfile(texts_or_paths):
            with open(texts_or_paths, 'r', encoding='utf-8') as f:
                corpus = f.read()
        elif isinstance(texts_or_paths, str) and not os.path.isfile(texts_or_paths):
            corpus = texts_or_paths
        elif isinstance(texts_or_paths, (list, tuple)):
            parts = []
            for p in texts_or_paths:
                if not isinstance(p, str) or not os.path.isfile(p):
                    raise ValueError(f"Expected file path string, got: {p!r}")
                with open(p, 'r', encoding='utf-8') as f:
                    parts.append(f.read())
            if not parts:
                raise ValueError("No files provided to build a corpus.")
            corpus = '\n'.join(parts)
        else:
            raise ValueError("Input should be a string (text or file path) or a list/tuple of file paths.")

        # I originally intended to make a manual loop through the corpus, but apparently Counter is already made for that.
        counts = Counter(corpus)
        items = [(ch, c) for ch, c in counts.items() if c >= min_freq]
        if not items and not include_specials:
            raise ValueError("No characters meet the frequency threshold; cannot build vocab.")

        # Sort by frequency (descending) and then alphabetically.
        items.sort(key=lambda x: (-x[1], x[0]))
        idx = 0
        # Add special tokens first if specified, then add the rest of the characters.
        self.stoi = {}
        self.pad_id = None
        self.unk_id = None

        if include_specials:
            self.stoi["<PAD>"] = idx
            self.pad_id = idx
            idx += 1
            self.stoi["<UNK>"] = idx
            self.unk_id = idx
            idx += 1
        for ch, _ in items:
            if ch not in self.stoi:
                self.stoi[ch] = idx
                idx += 1
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        #Optional internal sanity check (helps catch accidental mapping bugs early)
        if not self.stoi or not self.itos or any(self.itos[self.stoi[s]] != s for s in self.stoi):
            raise ValueError("Internal mapping inconsistency after fit().")
    
    def encode(self, text: str, strict: bool = False) -> List[int]: # Encode text to list of token IDs in stoi.
        if not self.is_fitted():
            raise RuntimeError("Tokenizer not fitted. Call fit(...) first.")
        if strict:
            return [self.stoi[ch] for ch in text]  # will raise KeyError automatically
        if self.unk_id is None:
            return [self.stoi[ch] for ch in text]  # will also raise KeyError on unknown
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, token_ids: List[int], skip_specials: bool = True) -> str:  # Decode list of token IDs back to text using itos.
        if not self.is_fitted():
            raise RuntimeError("Tokenizer not fitted. Call fit(...) first.")
        if skip_specials and self.pad_id is not None and self.unk_id is not None:
            return ''.join(self.itos[i] for i in token_ids if i != self.pad_id and i != self.unk_id)
        return ''.join(self.itos[i] for i in token_ids)

    def get_vocab_size(self) -> int:  # Return the size of the vocabulary.
        if not self.is_fitted():
            raise RuntimeError("Tokenizer not fitted. Call fit(...) first.")
        return self.vocab_size

    def save(self, path: str) -> None:
        """
        Serialize the tokenizer to a JSON file at `path`.
        Writes atomically (via a temp file + replace) to avoid partial files.
        """
        if not self.is_fitted():
            raise RuntimeError("Tokenizer not fitted. Call fit(...) before save().")

        payload = {
            "version": "char-tokenizer.v1",
            "stoi": self.stoi,          # dict[str, int]
            "pad_id": self.pad_id,      # may be None
            "unk_id": self.unk_id,      # may be None
        }

        # Ensure parent directory exists
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)

        # Atomic write: write to a temp file, then replace
        fd, tmp_path = tempfile.mkstemp(dir=parent or None, prefix=".tmp_tok_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            # On error, best-effort cleanup
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            finally:
                raise

    def load(self, path: str) -> None:
        """
        Load tokenizer state from a JSON file into THIS instance.
        Overwrites any existing state on the instance.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No such file: {path!r}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Basic validation
        if "stoi" not in data or not isinstance(data["stoi"], dict):
            raise ValueError("Invalid tokenizer file: missing or bad 'stoi'.")
        if "pad_id" not in data or "unk_id" not in data:
            raise ValueError("Invalid tokenizer file: missing 'pad_id'/'unk_id'.")

        self.stoi = {str(s): int(i) for s, i in data["stoi"].items()}
        self.pad_id = data["pad_id"] if data["pad_id"] is not None else None
        self.unk_id = data["unk_id"] if data["unk_id"] is not None else None

        # Rebuild inverse + size
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        # Sanity: mappings must invert
        if any(self.itos[self.stoi[s]] != s for s in self.stoi):
            raise ValueError("Loaded tokenizer has inconsistent stoi/itos.")

    @classmethod
    def from_file(cls, path: str) -> "CharTokenizer":
        """
        Construct a new CharTokenizer from a saved JSON file.
        """
        tok = cls()
        tok.load(path)
        return tok
