# MarkovText (robust term_count conversion, improved)
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
    """

    def __init__(self, corpus: str):
        if not isinstance(corpus, str):
            raise TypeError("corpus must be a string")
        self.corpus = corpus.strip()
        self.tokens: List[Token] = self._tokenize(self.corpus)
        self.term_dict_cache: Dict[int, Dict[State, List[Token]]] = {}
        self.term_dict = self.get_term_dict(k=1)

    @staticmethod
    def _tokenize(text: str) -> List[Token]:
        if not text:
            return []
        s = re.sub(r"\s+", " ", text).strip()
        s = re.sub(r"[\"“”\(\)\[\]\{\}:;,.!?<>«»]", "", s)
        tokens = [t.strip() for t in s.split(" ") if t.strip()]
        return tokens

    def get_term_dict(self, k: int = 1, use_cache: bool = True) -> Dict[State, List[Token]]:
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be an integer >= 1")
        if use_cache and k in self.term_dict_cache:
            return self.term_dict_cache[k]

        tokens = self.tokens
        term_dict_local: Dict[State, List[Token]] = defaultdict(list)
        n = len(tokens)
        if n == 0:
            self.term_dict_cache[k] = {}
            if k == 1:
                self.term_dict = {}
            return {}

        for i in range(n - k):
            state = tokens[i] if k == 1 else tuple(tokens[i : i + k])
            follower = tokens[i + k]
            term_dict_local[state].append(follower)

        term_dict_out = dict(term_dict_local)
        self.term_dict_cache[k] = term_dict_out
        if k == 1:
            self.term_dict = term_dict_out
        return term_dict_out

    # -------------------------
    # Text generation
    # -------------------------
    def generate(self, term_count: int = 50, seed_term: Union[str, Tuple[str, ...], None] = None, k: int = 1) -> str:
        """
        Generate text with exactly `term_count` tokens (if possible).
        """

        # ---- robust term_count conversion ----
        def _to_int_scalar(x):
            # Direct int or numpy integer
            try:
                if isinstance(x, (int, np.integer)):
                    return int(x)
                if isinstance(x, (float, np.floating)):
                    return int(x)
            except Exception:
                pass

            # numpy 0-d array
            try:
                if isinstance(x, np.ndarray) and x.shape == ():
                    return int(x.item())
            except Exception:
                pass

            # pandas single-value containers / scalars
            try:
                import pandas as pd  # local import to avoid hard dependency if not installed
                # pandas.Series or Index with single element
                if isinstance(x, pd.Series) and x.size == 1:
                    return int(x.iloc[0])
                if isinstance(x, (pd.Index, pd.Categorical)) and len(x) == 1:
                    return int(x[0])
                # pandas scalar types (e.g., pd.Int64Dtype scalar)
                if pd.api.types.is_scalar(x):
                    return int(x)
            except Exception:
                pass

            # objects implementing __int__
            try:
                if hasattr(x, "__int__"):
                    return int(x)
            except Exception:
                pass

            # try float conversion from string-like values
            try:
                return int(float(str(x)))
            except Exception:
                pass

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
                        raise ValueError(f"seed_term must contain exactly {k} tokens when k={k}")
                    candidate_state = tuple(parts)
                else:
                    raise ValueError("seed_term must be a tuple/list of tokens or a whitespace-separated string")

            if candidate_state not in term_dict:
                raise ValueError("seed_term not found in corpus states")
            state = candidate_state

        # initialize generated sequence
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
                    break
            next_token = np.random.choice(followers)
            generated.append(next_token)
            if k == 1:
                state = next_token
            else:
                state = tuple(list(state[1:]) + [next_token])

        return " ".join(generated)
