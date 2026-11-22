# Architecture

## Design goals

- Keep NLP algorithms independent from Streamlit so they can be tested and reused.
- Keep Streamlit rendering and session-state concerns out of algorithm services.
- Prefer deterministic, local algorithms; make optional model requirements visible.
- Treat validation failures as actionable messages rather than allowing tracebacks into the UI.
- Keep analysis history session-scoped and avoid sending user text to external services.

## Layers

```text
app.py
  ├── ui/components.py       Shared input, result, chart, and download components
  ├── ui/pages.py            Navigation and composition of the module views
  ├── modules/*.py           NLP service functions and algorithm-specific validation
  ├── utils/*.py             Text helpers, document extraction, model/resource loading
  ├── data/                  Small, documented, versioned demo datasets
  └── tests/                 Unit tests for services and app-level navigation checks
```

The existing module service files remain the public algorithm layer. The UI extraction is deliberately incremental: `app.py` remains the required `streamlit run app.py` entry point, while shared presentation logic moves behind `ui/`. Dataset examples belong in `data/`, rather than being mixed into classifier implementation.

## Module contract

Each NLP service accepts validated Python values and returns a small serializable result (dictionaries, lists, and numeric values). It should not import Streamlit, mutate global session state, write reports, or contact a remote API. Streamlit views own controls, charts, downloads, and user-facing error presentation.

## Runtime and model resources

Core algorithms use locally installed packages. spaCy resources are optional and loaded lazily; missing model packages select an explicitly named heuristic fallback. NLTK corpus-dependent operations must either use an installed corpus or a documented fallback. PDFs are parsed locally; scanned-image OCR is out of scope unless an OCR dependency is explicitly added.

## Testing strategy

- Unit tests cover normal results, invalid input, and at least one edge case per service.
- Deterministic random seeds are used for LDA and language detection.
- The Streamlit test harness verifies the dashboard and every navigation destination render.
- Tests should not require downloading language models or making network requests.
- Classification tests verify empty input, unknown tasks, duplicate/imbalanced data, reproducibility, and held-out metrics.

## Submission workflow

1. Create a Python 3.10+ virtual environment.
2. Install `requirements.txt`.
3. Optionally install `en_core_web_sm` for spaCy-based English NER/POS.
4. Run `streamlit run app.py`.
5. Load a TXT/PDF or enter text directly.
6. Open one of the 14 module views and run its action.
7. Download session reports from the Dashboard.

TXT files are UTF-8 decoded with a replacement policy. PDFs are parsed locally
with `pypdf`; encrypted PDFs and PDFs without extractable text are reported as
errors. OCR is intentionally not included.

## Current limitations

Several modules are educational baselines, not production NLP systems: the sample classifier has a small training set; TF-IDF QA is lexical retrieval; the FAQ chatbot is not a generative model; and rule-based NER/POS fallbacks are approximate. These distinctions should remain visible in both the interface and documentation.

Text classification uses a documented local benchmark and evaluates held-out
data with reproducible stratified splits. Model comparison does not imply
production readiness: the news benchmark is small, confidence-like scores are
uncalibrated, and Hindi classification is not tested or supported.
