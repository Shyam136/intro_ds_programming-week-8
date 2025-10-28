# MarkovText (robust term_count conversion)
from __future__ import annotations
from collections import defaultdict
import re
import numpy as np
from typing import List, Dict, Tuple, Union

Token = str
State = Union[Token, Tuple[Token, ...]]


class MarkovText:
    """
    Simple Markov-chain text generator.

    Parameters
    ----------
    corpus : str
        Text corpus (single string). Tokenization is whitespace-based with light punctuation stripping.

    Attributes
    ----------
    tokens : List[str]         # tokenized corpus
    term_dict : Dict[State, List[str]]   # first-order term dict (k=1)
    term_dict_cache : Dict[int, Dict]    # cached dicts for other k values
    """

    def __init__(self, corpus: str):
        if not isinstance(corpus, str):
            raise TypeError("corpus must be a string")
        self.corpus = corpus.strip()
        self.tokens: List[Token] = self._tokenize(self.corpus)
        # cache for k-specific dictionaries
        self.term_dict_cache: Dict[int, Dict[State, List[Token]]] = {}
        # build default (k=1) term_dict attribute so tests can access mt.term_dict
        self.term_dict = self.get_term_dict(k=1)

    # -------------------------
    # Tokenization
    # -------------------------
    @staticmethod
    def _tokenize(text: str) -> List[Token]:
        """
        Basic tokenization:
        - normalize whitespace
        - remove most punctuation except internal apostrophes/dashes
        - split on whitespace
        """
        if not text:
            return []
        s = re.sub(r"\s+", " ", text).strip()
        s = re.sub(r"[\"“”\(\)\[\]\{\}:;,.!?<>«»]", "", s)  # remove punctuation
        tokens = [t.strip() for t in s.split(" ") if t.strip()]
        return tokens

    # -------------------------
    # Build term dictionary
    # -------------------------
    def get_term_dict(self, k: int = 1, use_cache: bool = True) -> Dict[State, List[Token]]:
        """
        Build and return a term dictionary mapping states -> list of following tokens.

        Parameters
        ----------
        k : int
            Context size (k=1 => single-token state). For k>1, keys are tuples of length k.
        use_cache : bool
            If True, reuse cached dict for the same k.

        Returns
        -------
        dict: { state -> [followers...] } with duplicates included (empirical counts preserved)
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be an integer >= 1")

        if use_cache and k in self.term_dict_cache:
            return self.term_dict_cache[k]

        tokens = self.tokens
        term_dict_local: Dict[State, List[Token]] = defaultdict(list)
        n = len(tokens)
        if n == 0:
            self.term_dict_cache[k] = {}
            # if k==1, also set attribute
            if k == 1:
                self.term_dict = {}
            return {}

        # Build sliding windows: for i in [0 .. n-k-1], state=tokens[i:i+k], follower=tokens[i+k]
        for i in range(n - k):
            state = tokens[i] if k == 1 else tuple(tokens[i : i + k])
            follower = tokens[i + k]
            term_dict_local[state].append(follower)  # keep duplicates to preserve empirical distribution

        term_dict_out = dict(term_dict_local)
        self.term_dict_cache[k] = term_dict_out

        # keep first-order as attribute for convenience
        if k == 1:
            self.term_dict = term_dict_out

        return term_dict_out

    # -------------------------
    # Text generation
    # -------------------------
    def generate(self, term_count: int = 50, seed_term: Union[str, Tuple[str, ...], None] = None, k: int = 1) -> str:
        """
        Generate text using the k-order Markov chain.

        term_count : int
            TOTAL number of tokens to return (the final string will contain exactly `term_count` tokens).
        seed_term : str | tuple | None
            Optional seed state (see class doc for allowed formats).
        k : int
            Markov order (context length)
        """

        # ---- robust term_count conversion ----
        def _to_int_scalar(x):
            """Try several sensible conversions to an int scalar; raise ValueError if not possible."""
            # direct int()
            try:
                return int(x)
            except Exception:
                pass

            # float-like strings or floats
            try:
                return int(float(x))
            except Exception:
                pass

            # numpy or pandas scalar / 0-d array
            try:
                arr = np.asarray(x)
                if arr.size == 1:
                    try:
                        return int(arr.item())
                    except Exception:
                        # fallback for numpy types
                        try:
                            return int(arr.tolist())
                        except Exception:
                            pass
            except Exception:
                pass

            # last attempt: for pandas NA or similar, raise
            raise ValueError("term_count must be an integer or convertible to int")

        term_count = _to_int_scalar(term_count)

        if term_count <= 0:
            return ""

        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be an int >= 1")

        term_dict = self.get_term_dict(k=k)
        if not term_dict:
            return ""

        # build initial state
        if seed_term is None:
            state = np.random.choice(list(term_dict.keys()))
        else:
            # normalize seed_term for k==1 or k>1 (same logic as before)
            if k == 1:
                if not isinstance(seed_term, str):
                    raise ValueError("For k=1 seed_term must be a string token")
                candidate_state = seed_term
            else:
                if isinstance(seed_term, (tuple, list)):
                    if len(seed_term) != k:
                        raise ValueError(f"seed_term tuple/list must have length {k}")
                    candidate_state = tuple(str(x) for x in seed_term)
                elif isinstance(seed_term, str):
                    parts = seed_term.split()
                    if len(parts) != k:
                        raise ValueError(f"seed_term must contain exactly {k} tokens (space-separated) when k={k}")
                    candidate_state = tuple(parts)
                else:
                    raise ValueError("seed_term must be a tuple/list of tokens or a whitespace-separated string")

            if candidate_state not in term_dict:
                raise ValueError("seed_term not found in corpus states")
            state = candidate_state

        # initialize generated list to reflect starting state but ensure final length == term_count
        generated: List[str] = []
        if isinstance(state, tuple):
            generated.extend(list(state))
        else:
            generated.append(state)

        if len(generated) > term_count:
            raise ValueError("seed_term contains more tokens than term_count")

        # produce tokens until we reach exactly term_count length
        while len(generated) < term_count:
            followers = term_dict.get(state, [])
            if not followers:
                # fallback: pick a random existing state and continue
                state = np.random.choice(list(term_dict.keys()))
                followers = term_dict.get(state, [])
                if not followers:
                    # nothing useful in dict — break to avoid infinite loop
                    break
            next_token = np.random.choice(followers)
            generated.append(next_token)
            # advance state
            if k == 1:
                state = next_token
            else:
                state = tuple(list(state[1:]) + [next_token])

        return " ".join(generated)