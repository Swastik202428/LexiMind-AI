# NLP algorithms and sample checks

The suite uses small, reproducible examples so a demonstrator can explain the input, method, and expected output without internet access. Tests are in `tests/`; run them with `python -m unittest discover -s tests -v`.

| Module | Method | Sample check | Important limitation |
|---|---|---|---|
| Preprocessing | Configurable Unicode-aware normalization, stopword filtering, Porter stemming, optional WordNet lemmatization | `"The cats are running!"` with stemming yields `cat run`; punctuation removal preserves token boundaries | Suffix fallbacks are approximate; punctuation and tokenization policy affect results |
| Tokenization | Regex sentence boundaries, word tokens, and contiguous 1–3 grams | `"Cats chase mice"` includes the bigram `cats chase` | Abbreviations and language-specific sentence rules need a trained tokenizer |
| Statistics | Token counts, case-insensitive vocabulary, and `Counter` frequencies | `"One ONE two."` has 3 tokens and 2 types | Word rules are intentionally simple and Unicode-aware |
| Sentiment | TextBlob lexicon polarity and subjectivity | Clearly positive and negative sentences have corresponding signs | Sarcasm, domain, and context can reverse the apparent meaning |
| NER | spaCy `en_core_web_sm` when installed; pattern fallback otherwise | OpenAI, London, `$50`, and a written date are test examples | Fallback entity patterns are not general-purpose NER |
| Classification | TF-IDF 1–2 grams with Multinomial Naive Bayes, Logistic Regression, or Linear SVM baselines; reproducible stratified held-out evaluation | Prize-claim SMS → spam; football match → sports | Demonstration corpus only; displayed probability-like scores remain uncalibrated; Hindi classification is not supported or evaluated |
| Keywords | Sentence-level TF-IDF aggregation and corpus frequency | Repeated content terms rank above stopwords | A single sentence cannot provide document-level inverse-document-frequency evidence; the app labels this frequency fallback |
| Summarization | Sentence ranking by normalized content-word frequency, with deterministic source-order ties | Selected summary sentences must occur verbatim in source text | Extractive heuristic; sentence salience is not equivalent to factual importance |
| Language | `langdetect` probabilities with a very-short-text guard | An English paragraph returns `en`; short Devanagari returns `hi` by script | Mixed-language and short strings are uncertain |
| POS | spaCy model, NLTK tagger when data is installed, otherwise suffix/lexicon heuristic | A short sentence returns token/tag pairs | Rule-based fallback is not grammatical analysis |
| Lexical Vectors | Bag-of-Words, TF-IDF, and local window co-occurrence vectors | Vector rows align with document and feature counts | These are not pretrained or learned word embeddings; rename and UI messaging are explicit |
| Topic modeling | Scikit-learn LDA on a document-term count matrix | Multiple documents return the requested feasible topic count | Topics are sensitive to corpus size, vocabulary, and random initialization |
| Question answering | Porter-stemmed sentence-level TF-IDF cosine retrieval with a heuristic relevance acceptance threshold | Related inflections match; unrelated questions return no answer | Extractive baseline; it does not infer or generate missing facts; the cutoff is not validated on a benchmark |
| Chatbot | Intent shortcuts and TF-IDF similarity against a small FAQ set | Greeting, known NLP question, and unknown question have distinct intents | Fixed knowledge base; not a generative assistant |

Test cases focus on observable contracts and correctness. They intentionally avoid asserting implementation-only details such as private helper ordering or exact floating-point values where multiple library versions can differ.
