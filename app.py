"""Lab 03 - Chaotic Text Generator: a test bench for text generators (Streamlit UI).

Run:  streamlit run app.py
"""
import html
import json
import os
import random
import re

import altair as alt
import pandas as pd
import streamlit as st

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # data/ and results/ are relative to this file

from src.pipeline import HMM, NGramGenerator, coarse, generation_metrics, load_tagged, split_by_document  # noqa: E402
from src.suffix_hmm import SuffixHMM  # noqa: E402

st.set_page_config(page_title="Chaotic Text Generator", page_icon="🎲", layout="wide")

MODELS = ["HMM, plain states", "HMM, role-split states", "Trigram"]
COLORS = {"HMM, plain states": "#9aa5b1", "HMM, role-split states": "#2f6fdf", "Trigram": "#e8590c"}
NOTICE = ("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined "
          "relative to this corpus and do not measure the quality of real English.")
METRIC_HELP = {
    "valid_tag_pattern": "Tag sequence identical to one of the 16 tag sequences of the training set (structure).",
    "plausible_subject_verb": "The (first noun, first verb) pair occurs in the training set (meaning).",
    "novel_sentences": "Share of sentences that are NOT copied verbatim from the training set (memorisation).",
    "distinct_2": "Distinct word bigrams / all word bigrams of the batch (diversity).",
}
st.markdown("""
<style>
  .block-container {padding-top: 2rem;}
  .tok {display:inline-block;text-align:center;margin:4px 5px;padding:6px 12px;border-radius:10px;
        border:1px solid var(--c);background:color-mix(in srgb, var(--c) 8%, transparent);}
  .tok .w {font-size:1.1rem;} .tok .t {font-size:.78rem;font-weight:700;color:var(--c);letter-spacing:.03em;}
  .hero {padding:1.1rem 1.4rem;border-radius:14px;margin-bottom:.6rem;
         background:linear-gradient(120deg, rgba(47,111,223,.12), rgba(232,89,12,.10));}
  .hero h1 {margin:0;font-size:2.1rem;} .hero p {margin:.3rem 0 0;opacity:.8;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ models (estimated once)
@st.cache_resource
def get_models():
    train, test = split_by_document(load_tagged())
    models = {MODELS[0]: HMM(train), MODELS[1]: HMM(train, split_states=True),
              MODELS[2]: NGramGenerator([w for w, _ in train])}
    suffix = SuffixHMM(train, rare=10 ** 9, open_tags={"NOUN", "VERB", "ADJ"})  # tiny corpus: every word counts
    return models, suffix, train, test


models, suffix_tagger, train, test = get_models()
hmm_split = models[MODELS[1]]
train_text = {" ".join(w) for w, _ in train}
train_patterns = {" ".join(t) for _, t in train}


@st.cache_data(show_spinner=False)
def generate(model_name, temperature, seed, n):
    """One row per generated sentence, and the (words, coarse tags) samples used by the metrics."""
    rng = random.Random(int(seed))
    model = models[model_name]
    rows, samples = [], []
    for _ in range(n):
        if model_name == MODELS[2]:
            words = model.sample_sentence(rng, temperature)
            states = tags = hmm_split.viterbi(words)   # the trigram has no hidden states: tag it
        else:
            words, states = model.sample_with_states(rng, temperature)
            tags = coarse(states)
        text = " ".join(words)
        rows.append({"sentence": text, "hidden states": " ".join(states),
                     "valid pattern": " ".join(tags) in train_patterns,
                     "copied from training": text in train_text})
        samples.append((words, tags))
    return pd.DataFrame(rows), samples


@st.cache_data(show_spinner=False)
def metrics(model_name, temperature, seed, n):
    return generation_metrics(generate(model_name, temperature, seed, n)[1], train, hmm_split)


@st.cache_data(show_spinner="Sampling the three models at 9 temperatures…")
def temperature_sweep(seed, n):
    rows = []
    for name in MODELS:
        for t in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
            m = metrics(name, t, seed, n)
            rows.append({"model": name, "temperature": t, **m})
    return pd.DataFrame(rows)


@st.cache_data
def tagging_accuracy():
    out = {}
    for name in MODELS[:2]:
        ok = tot = 0
        for w, g in test:
            p = models[name].viterbi(w)
            ok += sum(a == b for a, b in zip(g, p)); tot += len(g)
        out[name] = ok / tot
    return out


def tokenize(text):
    return re.findall(r"[a-z]+|[.!?]", text.lower())


def model_scale():
    return alt.Scale(domain=MODELS, range=[COLORS[m] for m in MODELS])


# ------------------------------------------------------------------ header + sidebar
st.markdown("<div class='hero'><h1>🎲 Chaotic Text Generator</h1>"
            "<p>A test bench for text generators: a part-of-speech Hidden Markov Model (plain or role-split "
            "states) against a trigram model, with metrics on hundreds of sentences and a Viterbi tagger.</p></div>",
            unsafe_allow_html=True)
st.warning(NOTICE, icon="⚠️")

with st.sidebar:
    st.header("⚙️ Control panel")
    model_name = st.selectbox("Model", MODELS, index=1)
    temperature = st.slider("Temperature", 0.3, 3.0, 1.0, 0.1,
                            help="Sampling weights are count^(1/T). T < 1 is safer, T > 1 is more varied.")
    seed = int(st.number_input("Random seed", 0, 10_000, 42))
    n = st.slider("Number of sentences", 10, 500, 500, 10)
    go = st.button("🎲 Generate", type="primary", width="stretch")
    st.divider()
    st.caption(f"Training set: {len(train)} sentences · test set: {len(test)} sentences "
               f"(split by document) · {len(train_patterns)} distinct tag patterns · "
               f"{len(models[MODELS[0]].vocab)} words.")

tabs = st.tabs(["📝 Generate", "🏷️ Tagger", "📊 Compare models", "🌍 Real text", "🔍 Inside the model",
                "📖 How it works"])

# ------------------------------------------------------------------ 1. generate
with tabs[0]:
    if go:
        st.session_state["batch"] = (model_name, temperature, seed, n)
    if "batch" not in st.session_state:
        st.info("Choose a model and the settings in the control panel, then press **🎲 Generate**.")
    else:
        b_model, b_temp, b_seed, b_n = st.session_state["batch"]
        df, _ = generate(b_model, b_temp, b_seed, b_n)
        m = metrics(b_model, b_temp, b_seed, b_n)
        st.subheader(f"{b_model} · T = {b_temp} · seed {b_seed} · {b_n} sentences")
        c = st.columns(4)
        c[0].metric("Valid tag pattern", f"{m['valid_tag_pattern']:.1%}", help=METRIC_HELP["valid_tag_pattern"])
        c[1].metric("Plausible subject-verb", f"{m['plausible_subject_verb']:.1%}",
                    help=METRIC_HELP["plausible_subject_verb"])
        c[2].metric("Novelty", f"{m['novel_sentences']:.1%}", help=METRIC_HELP["novel_sentences"])
        c[3].metric("Distinct-2", f"{m['distinct_2']:.3f}", help=METRIC_HELP["distinct_2"])

        f1, f2 = st.columns([2, 1])
        query = f1.text_input("🔎 Filter sentences", placeholder="e.g. watch, garden, foxes …")
        show = f2.radio("Show", ["all", "invalid pattern only", "copied only"], horizontal=True)
        view = df.copy()
        if query:
            view = view[view["sentence"].str.contains(query.lower(), regex=False)]
        if show == "invalid pattern only":
            view = view[~view["valid pattern"]]
        elif show == "copied only":
            view = view[view["copied from training"]]
        view.index = view.index + 1
        st.dataframe(view, width="stretch", height=430, column_config={
            "sentence": st.column_config.TextColumn(width="large"),
            "hidden states": st.column_config.TextColumn(width="large"),
            "valid pattern": st.column_config.CheckboxColumn(help=METRIC_HELP["valid_tag_pattern"]),
            "copied from training": st.column_config.CheckboxColumn(help="Memorisation detector")})
        copied = int(df["copied from training"].sum())
        st.caption(f"Showing {len(view)} of {len(df)} sentences · "
                   f"{int((~df['valid pattern']).sum())} invalid patterns · "
                   f"{copied} copied from the training set (memorisation, not creativity)."
                   + (" · The trigram has no hidden states: the tags shown are Viterbi tags." if b_model == MODELS[2]
                      else ""))
        out = df.copy()
        out.insert(0, "model", b_model); out.insert(1, "temperature", b_temp); out.insert(2, "seed", b_seed)
        st.download_button("⬇️ Download CSV", out.to_csv(index_label="id").encode("utf-8"),
                           file_name=f"generated_{b_model.split(',')[0].lower().replace(' ', '_')}"
                                     f"{'_split' if 'split' in b_model else ''}_T{b_temp}_seed{b_seed}.csv",
                           mime="text/csv")

# ------------------------------------------------------------------ 2. tagger
with tabs[1]:
    st.subheader("Viterbi tagger")
    c1, c2 = st.columns([2, 1])
    tagger_name = c1.radio("Tagger", MODELS[:2], horizontal=True)
    use_suffix = c2.toggle("Guess unknown words from their suffix", value=False,
                           help="Extension: an unknown word gets P(tag | suffix) instead of the flat smoothed "
                                "emission. Uses plain states.")
    examples = ["the old farmers watch the birds in the garden .", "the children sell the watch .",
                "the hungry zebras eat the purple apples near the lake ."]
    ex = st.pills("Examples", examples, default=examples[0])
    text = st.text_input("Type a sentence", ex or "")
    tagger = suffix_tagger if use_suffix else models[tagger_name]
    words = tokenize(text)
    if not words:
        st.info("Type at least one word (lower-case English words from the vocabulary work best).")
    else:
        unknown = sorted({w for w in words if w not in tagger.vocab})
        if unknown:
            st.warning(f"Out-of-vocabulary word(s): **{', '.join(unknown)}**. Their tag is a guess"
                       + (" from the word suffix." if use_suffix else
                          " (smoothed emission only). Try the suffix toggle."), icon="⚠️")
        tags = tagger.viterbi(words)
        chips = []
        for w, t in zip(words, tags):
            color = "#d9480f" if w not in tagger.vocab else "#2f6fdf"
            chips.append(f"<div class='tok' style='--c:{color}'><div class='w'>{html.escape(w)}</div>"
                         f"<div class='t'>{t}{' ?' if w not in tagger.vocab else ''}</div></div>")
        st.markdown("".join(chips), unsafe_allow_html=True)
        st.caption("✅ This tag pattern is one of the training patterns." if " ".join(tags) in train_patterns
                   else "❌ This tag pattern does not appear in the training set.")
    with st.expander(f"Vocabulary ({len(models[MODELS[0]].vocab)} words)"):
        hmm0 = models[MODELS[0]]
        by_tag = {t: ", ".join(sorted(hmm0.emit[t])) for t in hmm0.tags}
        st.table(pd.DataFrame.from_dict(by_tag, orient="index", columns=["words"]))

# ------------------------------------------------------------------ 3. compare
with tabs[2]:
    st.subheader("Three models at the same settings")
    st.caption(f"T = {temperature}, seed = {seed}, {n} sentences per model (from the control panel).")
    table = pd.DataFrame([{"model": name, **metrics(name, temperature, seed, n)} for name in MODELS]).set_index("model")
    table = table[list(METRIC_HELP)].rename(columns=lambda k: k.replace("_", " "))
    st.dataframe(table.style.format({"valid tag pattern": "{:.1%}", "plausible subject verb": "{:.1%}",
                                     "novel sentences": "{:.1%}", "distinct 2": "{:.3f}"})
                 .highlight_max(color="rgba(47,111,223,.25)"), width="stretch")
    st.caption("Highlighted: best value of each column.")

    sweep = temperature_sweep(seed, n)
    metric_name = st.selectbox("Metric against temperature", list(METRIC_HELP), index=0,
                               format_func=lambda k: k.replace("_", " "))
    chart = (alt.Chart(sweep).mark_line(point=True, strokeWidth=2.5)
             .encode(x=alt.X("temperature:Q", title="temperature"),
                     y=alt.Y(f"{metric_name}:Q", title=metric_name.replace("_", " "),
                             axis=alt.Axis(format="%" if metric_name != "distinct_2" else ".2f")),
                     color=alt.Color("model:N", scale=model_scale(), legend=alt.Legend(orient="bottom")),
                     tooltip=["model", "temperature", alt.Tooltip(f"{metric_name}:Q", format=".3f")])
             .properties(height=340))
    st.altair_chart(chart, width="stretch")
    st.caption("Safety against diversity: as T rises the trigram makes more structural errors while its diversity "
               "grows. Role-split states keep 100% valid patterns at every temperature.")

# ------------------------------------------------------------------ 4. real text
with tabs[3]:
    st.subheader("The same HMM on real text: Universal Dependencies English EWT")
    try:
        with open("results/ewt_results.json", encoding="utf-8") as f:
            ewt = json.load(f)
    except (OSError, ValueError):
        ewt = None
        st.info("Run `python -m src.evaluate_real` to produce results/ewt_results.json.")
    if ewt:
        st.caption(f"{ewt.get('n_train', 12544):,} training sentences, {ewt.get('n_test', 2077):,} test sentences, "
                   f"17 UPOS tags, {ewt.get('distinct_tag_sequences', 10359):,} distinct tag sequences "
                   f"(synthetic corpus: 16). Words are lower-cased.")
        syn = tagging_accuracy()[MODELS[0]]
        rows = [{"model": "Synthetic corpus: HMM + Viterbi", "accuracy": syn, "known words": syn,
                 "unknown words": float("nan"), "whole sentence": 1.0}]
        for name, r in ewt["models"].items():
            rows.append({"model": f"EWT: {name}", "accuracy": r["accuracy"], "known words": r["known_words"],
                         "unknown words": r["unknown_words"], "whole sentence": r["whole_sentence"]})
        st.dataframe(pd.DataFrame(rows).set_index("model").style.format("{:.1%}", na_rep="—"), width="stretch")
        unk = ewt["models"]["HMM + Viterbi"]["unknown_share_of_tokens"]
        c = st.columns(3)
        c[0].metric("Unknown test tokens", f"{unk:.1%}")
        if "HMM + Viterbi + suffixes" in ewt["models"]:
            a, b = ewt["models"]["HMM + Viterbi"], ewt["models"]["HMM + Viterbi + suffixes"]
            c[1].metric("Unknown-word accuracy with suffixes", f"{b['unknown_words']:.1%}",
                        f"{(b['unknown_words'] - a['unknown_words']) * 100:+.1f} pts")
            c[2].metric("Overall accuracy with suffixes", f"{b['accuracy']:.1%}",
                        f"{(b['accuracy'] - a['accuracy']) * 100:+.1f} pts")
        keys = [("confusions", "HMM + Viterbi")] + ([("confusions_suffix", "HMM + Viterbi + suffixes")]
                                                     if "confusions_suffix" in ewt else [])
        conf = pd.DataFrame([{"error": f"{g} → {p}", "count": cnt, "model": label}
                             for key, label in keys for g, p, cnt in ewt[key][:10]])
        st.altair_chart(alt.Chart(conf).mark_bar().encode(
            y=alt.Y("error:N", sort="-x", title="gold → predicted"), x=alt.X("count:Q"),
            color=alt.Color("model:N", legend=alt.Legend(orient="bottom")), yOffset="model:N",
            tooltip=["model", "error", "count"]).properties(height=440, title="Most frequent tagging errors"),
            width="stretch")
        st.markdown(
            "**Reading.** The synthetic grammar is unambiguous once the previous tag is known, so 100% says nothing "
            "about English. On real text the HMM gains only 1.9 points over the most-frequent-tag baseline. "
            "Unknown words are the weak point: the smoothed emission k / (count(tag) + k·V) is largest for **rare** "
            "tags, so unseen words drift to X and INTJ. The suffix model (extension) estimates P(tag | suffix) on "
            "rare training words instead, which removes the PROPN → X / INTJ errors.")

# ------------------------------------------------------------------ 5. inside
with tabs[4]:
    st.subheader("Transition matrix P(next state | state)")
    which = st.radio("States", MODELS[:2], horizontal=True, key="heat")
    tags_, mat = models[which].transition_matrix()
    heat = pd.DataFrame([{"from": a, "to": b, "p": mat[i][j]} for i, a in enumerate(tags_)
                         for j, b in enumerate(tags_)])
    base = alt.Chart(heat).encode(x=alt.X("to:N", sort=tags_, title="next state"),
                                  y=alt.Y("from:N", sort=tags_, title="current state"))
    rect = base.mark_rect().encode(
        color=alt.Color("p:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), title="P"),
        tooltip=["from", "to", alt.Tooltip("p:Q", format=".2f")])
    text_ = base.mark_text(fontSize=11).encode(
        text=alt.condition("datum.p > 0.005", alt.Text("p:Q", format=".2f"), alt.value("")),
        color=alt.condition("datum.p > 0.6", alt.value("white"), alt.value("black")))
    st.altair_chart((rect + text_).properties(height=380 if which == MODELS[0] else 560), width="stretch")
    st.caption("Plain states: after NOUN the model cannot know whether the noun is the subject or the object "
               "(VERB 0.39, PREP 0.22, PUNCT 0.39). Role-split states: NOUN_S → VERB, NOUN_O → PREP or PUNCT, "
               "NOUN_P → PUNCT.")
    acc = tagging_accuracy()
    c = st.columns(2)
    for i, name in enumerate(MODELS[:2]):
        c[i].metric(f"Held-out tagging accuracy, {name}", f"{acc[name]:.1%}")

# ------------------------------------------------------------------ 6. how it works
with tabs[5]:
    st.subheader("How it works")
    st.markdown(r"""
**Hidden Markov Model.** The hidden states are part-of-speech tags and the observations are words. Three tables are
estimated by counting on the 960 training sentences (no training loop): start probabilities, transitions
$P(t_i \mid t_{i-1})$ and emissions $P(w_i \mid t_i)$.

**Generation.** Sample a tag sequence from the transitions, then one word per tag. The temperature reshapes the
sampling weights to $\text{count}^{1/T}$: $T<1$ sharpens (safe, repetitive), $T>1$ flattens (varied, more errors).

**Role-split states.** NOUN, ADJ and DET are renamed by their role: **S** before the verb, **O** after the verb,
**P** inside the prepositional phrase. A first-order model then *knows* where it is in the sentence.

**Viterbi.** Dynamic programming over positions and states, in **log space** (sums of log-probabilities, no underflow),
with add-k smoothing ($k = 0.01$) so that unseen events keep a non-zero probability.

**Trigram baseline.** $P(w_i \mid w_{i-2}, w_{i-1})$ with back-off to bigram and unigram.

**Metrics** (computed on the whole batch):
""")
    st.table(pd.DataFrame({"what it measures": ["structure", "meaning (approximation)", "memorisation", "diversity"],
                           "definition": list(METRIC_HELP.values())},
                          index=["valid tag pattern", "plausible subject-verb", "novelty", "distinct-2"]))
    st.markdown("""
**Limits and ethics.** The data is synthetic; the metrics are defined relative to its 16 tag patterns. A generator can
reproduce its training data: never train one on private messages you do not own. Generated text is fluent but has no
meaning or intent: do not present it as written by a person.
""")

st.divider()
st.caption("Lab 03 · NLP · Hidden Markov Model, Viterbi and trigram · "
           "[source code](https://github.com/hichamouaouche/hmm-text-generator)")
