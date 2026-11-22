"""Small offline text classifiers trained from included examples."""

import csv
import hashlib
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from utils.helpers import validate_text

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "classification_samples.csv"
TASKS = {"spam", "news"}


@lru_cache(maxsize=2)
def load_training_examples(task: str) -> list[tuple[str, str]]:
    """Read and validate the version-controlled task-specific dataset."""
    if task not in TASKS:
        raise ValueError("Choose either spam detection or news category.")
    try:
        with DATASET_PATH.open(encoding="utf-8", newline="") as dataset_file:
            reader = csv.DictReader(dataset_file)
            required_columns = {"task", "text", "label"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError("The classification sample CSV must contain task, text, and label columns.")
            all_examples = []
            for line_number, row in enumerate(reader, start=2):
                row_task = (row.get("task") or "").strip()
                text = (row.get("text") or "").strip()
                label = (row.get("label") or "").strip()
                if row_task not in TASKS or not text or not label:
                    raise ValueError(f"Invalid classification sample on CSV line {line_number}.")
                if row_task == task:
                    all_examples.append((text, label))
    except OSError as error:
        raise RuntimeError(f"Cannot read the sample classification dataset: {DATASET_PATH}") from error

    if len({label for _, label in all_examples}) < 2:
        raise ValueError(f"The sample dataset needs at least two labels for the {task!r} task.")
    validate_examples(all_examples)
    return all_examples


def validate_examples(examples: list[tuple[str, str]]) -> None:
    """Reject malformed, duplicate, or too-small evaluation corpora."""
    if len(examples) < 4:
        raise ValueError("At least four labeled examples are required for evaluation.")
    seen_texts = set()
    for text, label in examples:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Classification examples must contain non-empty text.")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Classification examples must contain non-empty labels.")
        fingerprint = hashlib.sha256(text.strip().casefold().encode("utf-8")).hexdigest()
        if fingerprint in seen_texts:
            raise ValueError("Duplicate text is not allowed in a classification dataset.")
        seen_texts.add(fingerprint)
    counts = {}
    for _, label in examples:
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2:
        raise ValueError("Classification evaluation needs at least two labels.")
    if min(counts.values()) < 5:
        raise ValueError("Each class needs at least five examples for a stratified 60/20/20 split.")


def _candidate_models():
    return {
        "TF-IDF + Multinomial Naive Bayes": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            MultinomialNB(),
        ),
        "TF-IDF + Logistic Regression": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(max_iter=1000, random_state=42),
        ),
        "TF-IDF + Linear SVM": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LinearSVC(random_state=42),
        ),
    }


def _metrics(labels, predictions, class_labels):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_labels,
        average=None,
        zero_division=0,
    )
    per_class = {
        str(label): {
            "precision": float(class_precision),
            "recall": float(class_recall),
            "macro_f1": float(class_f1),
            "support": int(class_support),
        }
        for label, class_precision, class_recall, class_f1, class_support in zip(
            class_labels, precision, recall, f1, support
        )
    }
    macro_precision = float(sum(precision) / len(precision)) if len(precision) else 0.0
    macro_recall = float(sum(recall) / len(recall)) if len(recall) else 0.0
    macro_f1 = float(sum(f1) / len(f1)) if len(f1) else 0.0
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_labels).tolist(),
        "class_labels": [str(label) for label in class_labels],
        "per_class": per_class,
    }


def evaluate_models(task: str, examples: list[tuple[str, str]] | None = None, random_state: int = 42) -> dict:
    """Compare TF-IDF baselines on a reproducible held-out test split."""
    if task not in TASKS:
        raise ValueError("Choose either spam detection or news category.")
    examples = load_training_examples(task) if examples is None else examples
    validate_examples(examples)

    texts = [text for text, _ in examples]
    labels = [label for _, label in examples]
    counts = {label: labels.count(label) for label in set(labels)}
    if any(count < 5 for count in counts.values()):
        raise ValueError("Each class needs at least five examples for a stratified 60/20/20 split.")

    train_texts, holdout_texts, train_labels, holdout_labels = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=random_state,
        stratify=labels,
    )
    fit_texts, validation_texts, fit_labels, validation_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.25,
        random_state=random_state,
        stratify=train_labels,
    )

    fingerprints = lambda items: {
        hashlib.sha256(text.strip().casefold().encode("utf-8")).hexdigest()
        for text in items
    }
    if fingerprints(fit_texts) & fingerprints(validation_texts):
        raise RuntimeError("Data leakage detected between training and validation data.")
    if fingerprints(fit_texts) & fingerprints(holdout_texts):
        raise RuntimeError("Data leakage detected between training and test data.")
    if fingerprints(validation_texts) & fingerprints(holdout_texts):
        raise RuntimeError("Data leakage detected between validation and test data.")

    class_labels = sorted(set(labels))
    model_results = {}
    selected_name = None
    selected_model = None
    selected_validation_f1 = -1.0

    for model_name, model in _candidate_models().items():
        model.fit(fit_texts, fit_labels)
        validation_predictions = model.predict(validation_texts)
        validation_metrics = _metrics(validation_labels, validation_predictions, class_labels)
        model_results[model_name] = {"validation": validation_metrics}
        if validation_metrics["macro_f1"] > selected_validation_f1:
            selected_name = model_name
            selected_validation_f1 = validation_metrics["macro_f1"]
            selected_model = model

    train_final_texts = fit_texts + validation_texts
    train_final_labels = fit_labels + validation_labels
    selected_model = _candidate_models()[selected_name]
    selected_model.fit(train_final_texts, train_final_labels)
    test_predictions = selected_model.predict(holdout_texts)
    test_metrics = _metrics(holdout_labels, test_predictions, class_labels)

    return {
        "task": task,
        "dataset_source": "LexiMind AI CC0-licensed local benchmark fixture",
        "random_state": random_state,
        "split": {
            "strategy": "stratified 60/20/20",
            "train": len(fit_texts),
            "validation": len(validation_texts),
            "test": len(holdout_texts),
        },
        "class_counts": {str(label): counts[label] for label in sorted(counts)},
        "selected_model": selected_name,
        "model_comparison": model_results,
        "test": test_metrics,
        "test_predictions": [
            {"text": text, "actual": str(actual), "predicted": str(predicted)}
            for text, actual, predicted in zip(holdout_texts, holdout_labels, test_predictions)
        ],
    }


@lru_cache(maxsize=2)
def get_evaluation(task: str) -> dict:
    return evaluate_models(task)


@lru_cache(maxsize=2)
def _model(task: str):
    examples = load_training_examples(task)
    texts, labels = zip(*examples)
    evaluation = get_evaluation(task)
    model = _candidate_models()[evaluation["selected_model"]]
    model.fit(texts, labels)
    return model


def classify_text(text: str, task: str = "spam") -> dict:
    validate_text(text)
    if task not in {"spam", "news"}:
        raise ValueError("Choose either spam detection or news category.")
    model = _model(task)
    prediction = str(model.predict([text])[0])
    classes = model.classes_
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
    else:
        import numpy as np

        decision = model.decision_function([text])[0]
        decision = np.asarray(decision).reshape(-1)
        if len(classes) == 2 and decision.size == 1:
            positive_probability = 1.0 / (1.0 + float(np.exp(-decision[0])))
            probabilities = np.array([1.0 - positive_probability, positive_probability])
        else:
            shifted = decision - np.max(decision)
            exponentials = np.exp(shifted)
            probabilities = exponentials / exponentials.sum()
    confidence = float(max(probabilities))
    return {
        "label": prediction,
        "confidence": confidence,
        "scores": {str(label): float(score) for label, score in zip(classes, probabilities)},
        "model": get_evaluation(task)["selected_model"],
        "training_examples": len(load_training_examples(task)),
        "evaluation": get_evaluation(task),
        "confidence_note": "Uncalibrated probability score; not a calibrated estimate.",
    }
