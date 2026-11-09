"""Lab 03 - Chaotic Text Generator: a test bench for text generators (Streamlit UI).

Run:  streamlit run app.py
"""
import os
import random
import re

import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import (HMM, NGramGenerator, coarse, generation_metrics, load_tagged,
                          split_by_document)

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/corpus.csv is relative to this file

st.set_page_config(page_title="Chaotic Text Generator", page_icon="🎲", layout="wide")

MODELS = ["HMM, plain states", "HMM, role-split states", "Trigram"]
NOTICE = ("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined "
          "relative to this corpus and do not measure the quality of real English.")


# ------------------------------------------------------------------ models (estimated once)
@st.cache_resource
def get_models():
    docs = load_tagged()
    train, test = split_by_document(docs)
    models = {
        MODELS[0]: HMM(train),
        MODELS[1]: HMM(train, split_states=True),
        MODELS[2]: NGramGenerator([w for w, _ in train]),
    }
    return models, train, test


models, train, test = get_models()
hmm_plain, hmm_split = models[MODELS[0]], models[MODELS[1]]
train_text = {" ".join(w) for w, _ in train}
train_patterns = {" ".join(t) for _, t in train}


def generate(model_name, temperature, seed, n):
    """Return a list of dicts (one per sentence) and the (words, coarse tags) samples for the metrics."""
    rng = random.Random(int(seed))
    model = models[model_name]
    rows, samples = [], []
    for _ in range(n):
        if model_name == MODELS[2]:
            words = model.sample_sentence(rng, temperature)
            states = hmm_split.viterbi(words)          # trigram has no hidden states: tag it with Viterbi
            tags = states
        else:
            words, states = model.sample_with_states(rng, temperature)
            tags = coarse(states)
        text = " ".join(words)
        rows.append({
            "sentence": text,
            "hidden states": " ".join(states),
            "valid tag pattern": "✅" if " ".join(tags) in train_patterns else "❌",
            "copied from training": "⚠️ copied" if text in train_text else "",
        })
        samples.append((words, tags))
    return rows, samples


def metrics_for(model_name, temperature, seed, n):
    _, samples = generate(model_name, temperature, seed, n)
    return generation_metrics(samples, train, hmm_split)


@st.cache_data
def comparison_table(temperature, seed, n):
    out = []
    for name in MODELS:
        m = metrics_for(name, temperature, seed, n)
        out.append({"model": name,
                    "valid tag pattern": f"{m['valid_tag_pattern']:.1%}",
                    "plausible subject-verb": f"{m['plausible_subject_verb']:.1%}",
                    "novel sentences": f"{m['novel_sentences']:.1%}",
                    "distinct-2": f"{m['distinct_2']:.3f}"})
    return pd.DataFrame(out).set_index("model")


@st.cache_data
def temperature_curve(seed, n):
    temps = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    data = {name: [metrics_for(name, t, seed, n)["valid_tag_pattern"] for t in temps] for name in MODELS}
    return pd.DataFrame(data, index=pd.Index(temps, name="temperature"))


def tokenize(text):
    return re.findall(r"[a-z]+|[.!?]", text.lower())


# ------------------------------------------------------------------ layout
st.title("🎲 Chaotic Text Generator")
st.caption("A test bench for text generators: POS-HMM (plain or role-split states) against a trigram model.")
st.warning(NOTICE, icon="⚠️")

with st.sidebar:
    st.header("Control panel")
    model_name = st.selectbox("Model", MODELS, index=1)
    temperature = st.slider("Temperature", 0.3, 3.0, 1.0, 0.1,
                            help="Sampling weights are count^(1/T): T < 1 is safer, T > 1 is more varied.")
    seed = st.number_input("Random seed", 0, 10_000, 42)
    n = st.slider("Number of sentences", 10, 500, 500, 10)
    go = st.button("Generate", type="primary", width="stretch")

tab_gen, tab_tag, tab_cmp, tab_model = st.tabs(["Generate", "Tagger (Viterbi)", "Compare models", "Model inside"])

# ---- 1. generation
with tab_gen:
    if go:
        st.session_state["batch"] = (model_name, temperature, int(seed), n)
    if "batch" not in st.session_state:
        st.info("Choose a model and the settings in the control panel, then press **Generate**.")
    else:
        b_model, b_temp, b_seed, b_n = st.session_state["batch"]
        rows, samples = generate(b_model, b_temp, b_seed, b_n)
        m = generation_metrics(samples, train, hmm_split)
        st.subheader(f"{b_model} · T = {b_temp} · seed = {b_seed} · {b_n} sentences")

        c = st.columns(4)
        c[0].metric("Valid tag pattern", f"{m['valid_tag_pattern']:.1%}",
                    help="Tag sequence identical to one of the 16 training sequences.")
        c[1].metric("Plausible subject-verb", f"{m['plausible_subject_verb']:.1%}",
                    help="(first noun, first verb) pair seen in the training set.")
        c[2].metric("Novelty", f"{m['novel_sentences']:.1%}",
                    help="Share of sentences not copied verbatim from the training set.")
        c[3].metric("Distinct-2", f"{m['distinct_2']:.3f}",
                    help="Distinct word bigrams / all word bigrams (diversity).")

        df = pd.DataFrame(rows)
        df.index = df.index + 1
        st.dataframe(df, width="stretch", height=420)
        copied = (df["copied from training"] != "").sum()
        if copied:
            st.caption(f"⚠️ {copied} sentence(s) are copied from the training set (memorisation, not creativity).")
        if b_model == MODELS[2]:
            st.caption("The trigram has no hidden states: the tags shown are the Viterbi tags of the generated words.")

        st.download_button("Download CSV", df.to_csv(index_label="id").encode("utf-8"),
                           file_name=f"generated_{b_model.split(',')[0].lower().replace(' ', '_')}"
                                     f"_T{b_temp}_seed{b_seed}.csv",
                           mime="text/csv")

# ---- 2. tagger
with tab_tag:
    st.subheader("Viterbi tagger")
    tagger_name = st.radio("Tagger", MODELS[:2], horizontal=True)
    text = st.text_input("Type a sentence", "the old farmers watch the birds in the garden .")
    tagger = models[tagger_name]
    words = tokenize(text)
    if not words:
        st.info("Type at least one word.")
    else:
        unknown = [w for w in words if w not in tagger.vocab]
        if unknown:
            st.warning(f"Out-of-vocabulary word(s): **{', '.join(sorted(set(unknown)))}**. "
                       "Their tag is a guess (smoothed emission probability only).", icon="⚠️")
        tags = tagger.viterbi(words)
        html = []
        for w, t in zip(words, tags):
            oov = w not in tagger.vocab
            color = "#d9534f" if oov else "#5b8def"
            html.append(
                f"<div style='display:inline-block;text-align:center;margin:4px 6px;padding:6px 10px;"
                f"border:1px solid {color};border-radius:8px'>"
                f"<div style='font-size:1.1rem'>{w}</div>"
                f"<div style='font-size:0.8rem;font-weight:600;color:{color}'>{t}{' ?' if oov else ''}</div></div>")
        st.markdown("".join(html), unsafe_allow_html=True)
        pattern_ok = " ".join(tags) in train_patterns
        st.caption(("✅ This tag pattern is one of the 16 training patterns."
                    if pattern_ok else "❌ This tag pattern does not appear in the training set."))
    with st.expander("Vocabulary"):
        st.write(", ".join(sorted(tagger.vocab)))

# ---- 3. comparison
with tab_cmp:
    st.subheader("Three models at the same settings")
    st.caption(f"T = {temperature}, seed = {int(seed)}, {n} sentences per model (from the control panel).")
    st.table(comparison_table(temperature, int(seed), n))
    st.subheader("Valid tag pattern against temperature")
    curve = temperature_curve(int(seed), n)
    st.line_chart(curve, x_label="temperature", y_label="valid tag pattern")
    st.caption("Role-split states keep 100% structure at every temperature; the trigram degrades as T rises.")

# ---- 4. inside the model
with tab_model:
    st.subheader("Transition matrix P(next tag | tag)")
    which = st.radio("States", MODELS[:2], horizontal=True, key="heat")
    tags_, mat = models[which].transition_matrix()
    heat = pd.DataFrame(np.array(mat), index=tags_, columns=tags_)
    st.dataframe(heat.style.background_gradient(cmap="Blues", vmin=0, vmax=1).format("{:.2f}"),
                 width="stretch")
    st.caption("With plain states, after NOUN the model cannot know if the noun is the subject or the object "
               "(VERB 0.39, PREP 0.22, PUNCT 0.39). Role-split states (S / O / P) give it that memory.")
    st.subheader("Held-out tagging accuracy")
    acc = {}
    for name in MODELS[:2]:
        ok = tot = 0
        for w, g in test:
            p = models[name].viterbi(w)
            ok += sum(a == b for a, b in zip(g, p)); tot += len(g)
        acc[name] = f"{ok / tot:.1%}"
    st.write(acc)

st.divider()
st.caption("Lab 03 · Hidden Markov Model + Viterbi vs trigram · synthetic corpus of 1,200 sentences, 6 tags.")
