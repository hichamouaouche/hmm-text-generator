# Lab 03: Chaotic Text Generator

A web test bench for text generators. It generates sentences with a part-of-speech Hidden Markov Model (plain or role-split states) or a trigram model, shows the hidden tag sequence behind each sentence, measures the batch with four metrics, and tags any typed sentence with the Viterbi algorithm.

**Live app:** `<add your Streamlit URL here>`

![Generate tab](screenshots/01_generate.png)

## Features

- **Control panel:** model (plain HMM, role-split HMM, trigram), temperature (0.3 to 3.0), random seed and number of sentences (10 to 500).
- **Generate:** a table with one row per sentence showing the words, the hidden states, whether the tag pattern is valid, and whether the sentence is copied from the training set.
- **Metrics:** valid tag pattern, plausible subject-verb, novelty and distinct-2, plus a **Download CSV** button.
- **Viterbi tagger:** shows a tag under each typed word. Out-of-vocabulary words are flagged with a warning, and the app does not crash.
- **Compare models:** the three models at the same settings, and a plot of valid tag pattern against temperature.
- **Model inside:** a heat-map of the transition matrix and the held-out tagging accuracy.

## Results (500 sentences, seed 42, T = 1.0)

| Model | Valid tag pattern | Plausible subject-verb | Novel | Distinct-2 |
|---|---|---|---|---|
| HMM, plain states | 16.2% | 14.8% | 99.8% | 0.198 |
| HMM, role-split states | 100% | 60.6% | 99.4% | 0.148 |
| Trigram | 57.0% | 76.0% | 91.2% | 0.105 |

On real text (UD English EWT), the same HMM with Viterbi tagging reaches 88.1% accuracy, against 86.2% for the most-frequent-tag baseline. It gets 28.2% of unknown words right and 40.4% of whole sentences. See [report/report.pdf](report/report.pdf).

## Run

```
pip install -r requirements.txt
python -m src.evaluate          # synthetic corpus -> results/metrics.json, results/hmm_results.png, results/failures.txt
python -m src.evaluate_real     # real text -> results/ewt_results.json (downloads UD English EWT on first run)
streamlit run app.py
```

## Files

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `src/pipeline.py` | HMM, Viterbi, temperature sampling, role-split states, trigram generator, metrics |
| `src/evaluate.py` / `src/evaluate_real.py` | Evaluation scripts for the synthetic corpus and for real text |
| `data/corpus.csv` | Synthetic tagged corpus: 1,200 sentences, 6 tags (required by the app) |
| `results/` | Outputs of the evaluation scripts |
| `report/` | One-page report (HTML source and PDF) |
| `screenshots/` | Screenshots of the application |

## Deploy (Streamlit Community Cloud)

Push this folder to GitHub, including `requirements.txt` and `data/corpus.csv`. On share.streamlit.io, choose **New app**, then select the repository and `app.py`. Do not commit `data/ewt/`; it is listed in `.gitignore`.

## Data and limits

`data/corpus.csv` is synthetic: a tiny hand-written grammar produced it (seed 42). The valid-tag-pattern and plausible-subject-verb metrics are defined relative to this corpus and do not measure the quality of real English. UD English EWT is licensed CC BY-SA 4.0 (https://github.com/UniversalDependencies/UD_English-EWT).
