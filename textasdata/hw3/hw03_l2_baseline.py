from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import random
import re
import time
import warnings

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

MODEL_NAME = "l2_baseline"
ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pairs(path):
    """Reject malformed data instead of silently dropping or relabeling rows."""
    with Path(path).open(encoding="utf-8-sig") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty list in {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"Row {index} must be [text, label] in {path}")
        if not isinstance(row[0], str) or not row[0].strip():
            raise ValueError(f"Empty/non-string text in row {index} of {path}")
        if type(row[1]) is not int or row[1] not in (0, 1):
            raise ValueError(f"Expected integer label 0 or 1 in row {index}")
    if {row[1] for row in rows} != {0, 1}:
        raise ValueError(f"Both NEG=0 and CORE=1 must be present in {path}")
    return [row[0] for row in rows], np.array([row[1] for row in rows])


def make_classifier():
    """Use the installed sklearn version's supported penalty syntax."""
    major_minor = tuple(int(v) for v in re.match(r"(\d+)\.(\d+)", sklearn.__version__).groups())
    regularization = ({"l1_ratio": 0.0} if major_minor >= (1, 8)
                      else {"penalty": "l2"})
    return LogisticRegression(
        **regularization,
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
        tol=1e-4,
        random_state=42,
    )


def review_sample(y_true, y_pred, seed=42):
    """Up to ten errors: five FP and five FN, filling from the other type.

    Sampling is fixed before reading cases; it does not select persuasive
    examples based on their content or model confidence.
    """
    fp = np.flatnonzero((y_true == 0) & (y_pred == 1)).tolist()
    fn = np.flatnonzero((y_true == 1) & (y_pred == 0)).tolist()
    rng = random.Random(seed)
    selected = rng.sample(fp, min(5, len(fp))) + rng.sample(fn, min(5, len(fn)))
    remaining = sorted(set(fp + fn) - set(selected))
    selected += rng.sample(remaining, min(10 - len(selected), len(remaining)))
    return sorted(selected)


def save_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / MODEL_NAME)
    args = parser.parse_args()
    train_file = args.data_dir / "train_core_vs_neg.json"
    test_file = args.data_dir / "test_core_vs_neg.json"
    for path in (train_file, test_file):
        if not path.is_file():
            parser.error(f"Missing {path}. Supply the saved Week 09 JSON files; "
                         "this model script will not invent data or recreate a split.")
    X_train_texts, y_train = load_pairs(train_file)
    X_test_texts, y_test = load_pairs(test_file)

    # Exactly the Week 09 settings; all other vectorizer options stay at defaults.
    vectorizer = TfidfVectorizer(lowercase=True, min_df=5, max_df=0.9)
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)
    clf = make_classifier()
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_train, y_train)
    convergence = [str(w.message) for w in caught
                   if issubclass(w.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError("Model did not converge; do not report these outputs: "
                           + "; ".join(convergence))
    elapsed = time.perf_counter() - started
    assert clf.classes_.tolist() == [0, 1]
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    assert np.isfinite(y_prob).all()
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, labels=[0, 1],
                                   target_names=["NEG", "CORE"],
                                   output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, labels=[0, 1],
                                        target_names=["NEG", "CORE"],
                                        digits=6, zero_division=0)
    auc = float(roc_auc_score(y_test, y_prob))
    names = vectorizer.get_feature_names_out()
    coefficients = clf.coef_[0]
    positive = sorted(np.flatnonzero(coefficients > 0),
                      key=lambda i: (-coefficients[i], names[i]))[:15]
    negative = sorted(np.flatnonzero(coefficients < 0),
                      key=lambda i: (coefficients[i], names[i]))[:15]
    nonzero = int(np.count_nonzero(coefficients))  # Excludes the intercept.
    assert cm.sum() == len(y_test)
    samples = review_sample(y_test, y_pred)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": MODEL_NAME, "train_size": len(y_train), "test_size": len(y_test),
        "train_class_counts": {str(k): int((y_train == k).sum()) for k in (0, 1)},
        "test_class_counts": {str(k): int((y_test == k).sum()) for k in (0, 1)},
        "features": len(names), "nonzero_coefficients": nonzero,
        "positive_coefficients": int((coefficients > 0).sum()),
        "negative_coefficients": int((coefficients < 0).sum()),
        "zero_fraction": float((coefficients == 0).mean()),
        "confusion_matrix": cm.tolist(), "classification_report": report,
        "roc_auc": auc, "intercept": float(clf.intercept_[0]),
        "iterations": clf.n_iter_.tolist(), "fit_seconds": elapsed,
        "convergence_warnings": convergence,
        "train_sha256": sha256(train_file), "test_sha256": sha256(test_file),
        "vocabulary_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "idf_sha256": hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest(),
        "vectorizer_settings": {"lowercase": True, "min_df": 5, "max_df": 0.9,
                                "other_parameters": "scikit-learn defaults"},
        "classifier_settings": clf.get_params(),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "error_review_test_indices": samples,
        "evaluation_target": "Agreement with weak labels, not independently verified historical truth",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    coefficient_rows = [{"word": str(name), "coefficient": float(value)}
                        for name, value in zip(names, coefficients)]
    save_csv(out / "coefficients.csv", coefficient_rows, ["word", "coefficient"])
    top_rows = [{"direction": direction, "rank": rank, "word": str(names[i]),
                 "coefficient": float(coefficients[i])}
                for direction, indices in [("CORE", positive), ("NEG", negative)]
                for rank, i in enumerate(indices, 1)]
    save_csv(out / "top_words.csv", top_rows, ["direction", "rank", "word", "coefficient"])
    predictions = [{"test_index": i, "weak_label": int(y_test[i]),
                    "prediction": int(y_pred[i]), "p_core": float(y_prob[i]),
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
                   for i, text in enumerate(X_test_texts)]
    save_csv(out / "predictions.csv", predictions,
             ["test_index", "weak_label", "prediction", "p_core", "text_sha256"])
    reviews = [{**predictions[i], "error_type": "FP" if y_test[i] == 0 else "FN",
                "text": X_test_texts[i]} for i in samples]
    (out / "error_review_sample.json").write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, out / "model.joblib")

    lines = [f"{MODEL_NAME} | scikit-learn {sklearn.__version__}",
             f"Train={len(y_train)}; Test={len(y_test)}; Features={len(names)}",
             f"Train SHA256: {metrics['train_sha256']}",
             f"Test SHA256:  {metrics['test_sha256']}",
             "\nConfusion matrix: rows=weak labels, columns=predictions; order=[NEG, CORE]",
             str(cm), "\nClassification report:", report_text, f"ROC AUC: {auc:.9f}",
             f"Non-zero coefficients: {nonzero} / {len(names)} (intercept excluded)",
             f"Iterations: {clf.n_iter_.tolist()}; no convergence warning"]
    for direction, indices in [("positive / CORE", positive), ("negative / NEG", negative)]:
        lines.append(f"\nTop {len(indices)} strictly {direction} words:")
        lines.extend(f"{names[i]:24s} {coefficients[i]:+.9f}" for i in indices)
        if len(indices) < 15:
            lines.append("Fewer than 15 non-zero coefficients of this sign; zeros are not added.")
    lines.append(f"\nFixed review sample: {len(reviews)} errors (at most 10).")
    for row in reviews:
        lines.extend([f"\nTest index {row['test_index']} | {row['error_type']} | "
                      f"weak_label={row['weak_label']} prediction={row['prediction']} "
                      f"P(CORE)={row['p_core']:.6f}", row['text']])
    console = "\n".join(lines)
    print(console)
    (out / "console_output.txt").write_text(console + "\n", encoding="utf-8")
    print(f"\nSaved outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
