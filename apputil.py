# apputil.py
from __future__ import annotations
from collections import defaultdict
import re
import numpy as np
from typing import List, Dict, Tuple, Union, Iterable

Token = str
State = Union[Token, Tuple[Token, ...]]


class MarkovText:
    """
    Simple Markov-chain text generator.

    Parameters
    ----------
    corpus : str
        Text corpus (single string). Tokenization is whitespace-based with light punctuation stripping.
    """

    def __init__(self, corpus: str):
        if not isinstance(corpus, str):
            raise TypeError("corpus must be a string")
        self.corpus = corpus.strip()
        self._tokens: List[Token] = self._tokenize(self.corpus)
        self.term_dict_cache: Dict[int, Dict[State, List[Token]]] = {}

    # -------------------------
    # Tokenization
    # -------------------------
    @staticmethod
    def _tokenize(text: str) -> List[Token]:
        """
        Basic tokenization:
        - normalize whitespace
        - remove most punctuation except internal apostrophes (e.g., don't)
        - split on whitespace
        """
        # normalize whitespace
        s = re.sub(r"\s+", " ", text).strip()
        # remove punctuation except apostrophe and dash (keep contractions)
        s = re.sub(r"[\"“”(),.:;!?—\[\]{}<>«»]", "", s)
        # split
        tokens = s.split(" ")
        # remove empty tokens and strip
        tokens = [t.strip() for t in tokens if t.strip()]
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

        tokens = self._tokens
        term_dict: Dict[State, List[Token]] = defaultdict(list)

        # Build sliding window
        n = len(tokens)
        if n == 0:
            self.term_dict_cache[k] = {}
            return {}

        # iterate through tokens; for index i, state is tokens[i:i+k], follower is tokens[i+k]
        for i in range(n - k):
            if k == 1:
                state = tokens[i]
            else:
                state = tuple(tokens[i:i + k])
            follower = tokens[i + k]
            term_dict[state].append(follower)  # include duplicates to preserve frequency

        # Optionally store terminal state with no follower? We'll leave absent; handle in generate().
        # Cache and return
        self.term_dict_cache[k] = dict(term_dict)
        return self.term_dict_cache[k]

    # -------------------------
    # Text generation
    # -------------------------
    def generate(self, term_count: int = 50, seed_term: str | None = None, k: int = 1) -> str:
        """
        Generate text using the k-order Markov chain.

        Parameters
        ----------
        term_count : int
            Number of tokens to generate (not including initial seed if provided).
        seed_term : str or None
            Optional starting token (for k=1) or space-separated k-token string (for k>1).
            If provided and not found in the corpus, raises ValueError.
        k : int
            Markov order (context length).

        Returns
        -------
        Generated text string
        """
        if term_count <= 0:
            return ""

        term_dict = self.get_term_dict(k=k)
        if not term_dict:
            return ""

        # Build initial state
        if seed_term is None:
            # choose random state from available keys
            state = np.random.choice(list(term_dict.keys()))
        else:
            # parse seed for k
            if k == 1:
                state_candidate: State = seed_term
            else:
                parts = seed_term.split()
                if len(parts) != k:
                    raise ValueError(f"seed_term must contain exactly {k} tokens (space separated) when k={k}")
                state_candidate = tuple(parts)

            # validate presence
            if state_candidate not in term_dict:
                raise ValueError("seed_term not found in corpus states")
            state = state_candidate

        generated: List[str] = []
        # If initial state is a tuple, extend generated with its tokens (so output starts sensibly)
        if isinstance(state, tuple):
            generated.extend(list(state))
        else:
            generated.append(state)

        # now iteratively sample followers
        for _ in range(term_count):
            followers = term_dict.get(state, [])
            if not followers:
                # no followers for this state; stop early
                break
            # sample according to empirical frequency
            next_token = np.random.choice(followers)
            generated.append(next_token)

            # advance state
            if k == 1:
                state = next_token
            else:
                # shift tuple left and append next_token
                state = tuple(list(state[1:]) + [next_token])

        return " ".join(generated)