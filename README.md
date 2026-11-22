# LexiMind AI

**Understand Language. Unlock Intelligence.**

LexiMind AI is a local-first Natural Language Processing workbench built with Python and Streamlit. It brings 14 explainable NLP demonstrations into one interface, from text normalization and statistical analysis to topic modeling and extractive question answering.

## Features

- English and Hindi-aware Unicode preprocessing
- English/Hindi tokenization and 1–3 grams
- Text statistics and corrected Hindi word frequencies
- TextBlob sentiment analysis
- spaCy NER/POS with visible local fallbacks
- TF-IDF classical-model spam and news classification
- Keyword extraction and extractive summarization
- Language detection, lexical vectors, LDA topic modeling
- Extractive question answering and an FAQ chatbot
- TXT and PDF text import
- Session history and TXT/CSV report export

## Run locally

Python 3.10 or newer is recommended.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

From the project directory, open `http://localhost:8501` after Streamlit starts.

The application has no paid API or API-key dependency. Most features run
offline after installation.

spaCy's optional English model:

```powershell
python -m spacy download en_core_web_sm
```

Without that model, NER/POS use visible fallback implementations. WordNet is
optional and English-only; preprocessing uses a suffix fallback when it is
missing. The project also uses a conservative built-in Hindi suffix fallback,
not a production Hindi morphological analyzer.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The suite uses standard `unittest` and Streamlit's app test harness.

## Project structure

```text
app.py                  Streamlit entry point and navigation
modules/                UI-independent NLP algorithms
utils/                  Shared text, session, and document helpers
ui/                     Shared Streamlit components and page composition
data/                   Versioned sample datasets
tests/                  Unit and app-navigation tests
docs/                   Architecture and algorithm notes
style.css               Application theme
requirements.txt        Runtime dependencies
```

## Submission status

LexiMind AI is a college-level local NLP demonstration, not a production NLP
service. Results from small educational datasets, heuristics, and fallbacks
must be interpreted accordingly.

### Text classification

The news fixture is a small, original CC0 1.0 local benchmark. It is not a
replacement for a representative external news benchmark. Held-out metrics and
confusion matrices are shown in the Text Classification module.

Current held-out news test metrics:

- Accuracy: 50.00%
- Macro precision: 46.00%
- Macro recall: 50.00%
- Macro-F1: 44.13%

Current held-out spam test metrics:

- Accuracy: 83.33%
- Macro precision: 87.50%
- Macro recall: 83.33%
- Macro-F1: 82.86%

These values are reproducible with random state 42 on the 60/20/20 split.
They are not production guarantees.

### Lexical vectors

The module named **Lexical Vectors** provides Bag-of-Words, TF-IDF, and local
context-cooccurrence vectors. It does not load pretrained word embeddings.

## Known limitations

- Scanned PDFs need OCR, which is not implemented.
- Hindi support is partial; preprocessing and tokenization are improved, but
  dedicated Hindi POS/NER, sentiment, and LLM resources are not included.
- spaCy's English model is not a Hindi NLP model.
- Hindi stemming is rule-based.
- TextBlob sentiment is English-oriented.
- Summarization and question answering are extractive baselines.
- Topic modeling is sensitive to small document collections.
- The FAQ chatbot uses fixed responses and TF-IDF similarity.
- Classification confidence and probability-like scores are uncalibrated.
- A PDF can be read only if it contains extractable text.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the architecture and [docs/ALGORITHMS.md](docs/ALGORITHMS.md) for method descriptions, test fixtures, and limitations.
