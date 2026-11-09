"""Unit tests of the model code: the numbers of the lab guide must be reproduced exactly."""
import random

import pytest

from src.pipeline import HMM, NGramGenerator, generation_metrics, load_tagged, refine, split_by_document
from src.suffix_hmm import SuffixHMM


@pytest.fixture(scope="module")
def data():
    train, test = split_by_document(load_tagged())
    return train, test, HMM(train), HMM(train, split_states=True), NGramGenerator([w for w, _ in train])


def test_split_sizes(data):
    train, test, *_ = data
    assert (len(train), len(test)) == (960, 240)
    assert len({" ".join(t) for _, t in train}) == 16


def test_refine_roles():
    assert refine("DET NOUN VERB DET NOUN PREP DET NOUN PUNCT".split()) == \
        "DET_S NOUN_S VERB DET_O NOUN_O PREP DET_P NOUN_P PUNCT".split()


def test_viterbi_accuracy_and_watch(data):
    _, test, hmm, hmm2, _ = data
    for model in (hmm, hmm2):
        pairs = [(g, p) for w, gold in test for g, p in zip(gold, model.viterbi(w))]
        assert sum(g == p for g, p in pairs) == len(pairs)
    assert hmm.viterbi("the children sell the watch .".split())[4] == "NOUN"
    assert hmm.viterbi("the children watch the birds .".split())[2] == "VERB"


@pytest.mark.parametrize("name, expected", [
    ("hmm", (0.162, 0.148, 0.998, 0.198)),
    ("hmm_split", (1.0, 0.606, 0.994, 0.148)),
    ("ngram", (0.570, 0.760, 0.912, 0.105)),
])
def test_generation_metrics_at_T1(data, name, expected):
    train, _, hmm, hmm2, ng = data
    rng = random.Random(42)
    if name == "ngram":
        samples = [(ng.sample_sentence(rng, 1.0), None) for _ in range(500)]
        m = generation_metrics(samples, train, hmm2)
    else:
        model = hmm if name == "hmm" else hmm2
        m = generation_metrics([model.sample_sentence(rng, 1.0) for _ in range(500)], train, hmm)
    got = (m["valid_tag_pattern"], m["plausible_subject_verb"], m["novel_sentences"], m["distinct_2"])
    assert got == pytest.approx(expected, abs=1e-3)


def test_unknown_words_do_not_crash(data):
    _, _, hmm, hmm2, _ = data
    for model in (hmm, hmm2):
        tags = model.viterbi(["qwerty"])
        assert len(tags) == 1
        assert len(model.viterbi("the zebras eat the qwerty .".split())) == 6


def test_suffix_model_keeps_known_words_and_guesses_unknown(data):
    train, test, hmm, *_ = data
    suf = SuffixHMM(train, rare=10 ** 9, open_tags={"NOUN", "VERB", "ADJ"})
    for w, gold in test[:50]:
        assert suf.viterbi(w) == gold
    assert suf.viterbi("the zebras eat the cakes .".split())[1] == "NOUN"
