# Included classification examples

`classification_samples.csv` contains separate local datasets for spam and news-topic classification.

The spam rows remain a small educational binary benchmark. The news rows are a
representative, project-curated English benchmark covering Politics, Business,
Science, Sports, and Entertainment. This replacement dataset is licensed under
CC0 1.0 as an original LexiMind AI test fixture.

The `task` column selects `spam` or `news`; `text` is the input feature; `label`
is the target class. Metrics are computed only from held-out examples and are
not derived from training accuracy.
