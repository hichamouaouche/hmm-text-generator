"""Extension: better unknown words in Viterbi with a suffix model (guide, section 5).

The plain HMM gives an unseen word the smoothed emission k / (count(tag) + k*V). That value is
largest for RARE tags, so unknown words drift to X or INTJ. Here an unknown word w receives instead

    P(w | t)  proportional to  P(t | suffix(w)) / P(t)                      (Bayes, P(w) is constant)

where P(t | suffix) is estimated on RARE training words (frequency <= `rare`), which behave like
unseen words, for every suffix of length 1..`max_suffix`, and interpolated from short to long
suffixes (as in the TnT tagger). Simple word-shape rules handle numbers and punctuation.
Known words keep exactly the emission of the base HMM.
"""
import math
import re
from collections import Counter, defaultdict

from src.pipeline import HMM

_NUM = re.compile(r"^[\d.,:/-]*\d[\d.,:/-]*$")
_PUNCT = re.compile(r"^[^\w\s]+$")


class SuffixHMM(HMM):
    def __init__(self, tagged_sentences, k=0.01, split_states=False, max_suffix=4, rare=10, open_tags=None):
        super().__init__(tagged_sentences, k=k, split_states=split_states)
        self.max_suffix = max_suffix
        self.open_tags = set(open_tags) if open_tags else None   # tags an unseen word may take
        word_freq = Counter(w for c in self.emit.values() for w, n in c.items() for _ in range(n))
        self.suffix = defaultdict(Counter)            # suffix -> Counter(tag)
        rare_tags = Counter()
        for t, c in self.emit.items():
            for w, n in c.items():
                if word_freq[w] <= rare:
                    rare_tags[t] += n
                    for L in range(0, min(max_suffix, len(w)) + 1):
                        self.suffix[w[len(w) - L:] if L else ""][t] += n
        total = sum(rare_tags.values()) or 1
        # prior over tags for unknown words: open-class tags dominate among rare words
        self.p_tag_rare = {t: (rare_tags[t] + 0.5) / (total + 0.5 * len(self.tags)) for t in self.tags}
        self.p_tag = {t: self.tag_totals[t] / sum(self.tag_totals.values()) for t in self.tags}
        # interpolation weight theta = std. dev. of the tag prior (TnT)
        mean = 1 / len(self.tags)
        self.theta = math.sqrt(sum((p - mean) ** 2 for p in self.p_tag_rare.values()) / len(self.tags))
        self._cache = {}

    def tag_given_suffix(self, w):
        """P(tag | word) for an unknown word, interpolated over its suffixes (short -> long)."""
        if w in self._cache:
            return self._cache[w]
        probs = dict(self.p_tag_rare)
        for L in range(1, min(self.max_suffix, len(w)) + 1):
            c = self.suffix.get(w[-L:])
            if not c:
                break
            n = sum(c.values())
            probs = {t: (c[t] / n + self.theta * probs[t]) / (1 + self.theta) for t in self.tags}
        if _NUM.match(w) and "NUM" in probs:
            probs = {t: (0.95 if t == "NUM" else 0.05 * p) for t, p in probs.items()}
        elif _PUNCT.match(w) and ("PUNCT" in probs or "SYM" in probs):
            probs = {t: (0.95 if t in ("PUNCT", "SYM") else 0.05 * p) for t, p in probs.items()}
        if self.open_tags:                           # closed classes (DET, PREP, ...) do not get new words
            probs = {t: (p if t in self.open_tags else 1e-6 * p) for t, p in probs.items()}
        self._cache[w] = probs
        return probs

    def p_emit(self, t, w):
        if w in self.vocab:
            return super().p_emit(t, w)
        p = self.tag_given_suffix(w).get(t, 1e-12) / max(self.p_tag[t], 1e-12)
        return max(p, 1e-12) * 1e-4                    # constant scale: unknown word, same for all tags

    def guess(self, w):
        """Most likely tag of an unknown word from its suffix alone (for display in the app)."""
        probs = self.tag_given_suffix(w)
        return max(probs, key=probs.get)
