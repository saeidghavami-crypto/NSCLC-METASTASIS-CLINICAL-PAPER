#!/usr/bin/env python3
"""Permutation-calibrated exploratory learning for paired NSCLC lipidomics.

The biological unit is the patient. For each lipid or prespecified lipid class,
the model uses the within-patient paired change:

    delta = log2(lymph-node abundance / matched primary-tumour abundance)

The analysis is deliberately designed for a five-patient pilot cohort. It does
not claim external predictive validity or biomarker validation. It uses:
  * a prespecified 25-lipid-class representation as the primary input;
  * L2-penalized logistic regression with scaling learned inside each
    leave-one-patient-out (LOPO) training fold;
  * exhaustive evaluation of all 10 possible 3-vs-2 class assignments;
  * sensitivity analyses across feature representations and regularization;
  * leave-one-patient-out coefficient stability;
  * an exact distance-based multivariate permutation test.

Dependencies: numpy, scipy, scikit-learn, matplotlib, Pillow.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


PRIMARY_C = 0.1
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
PATIENTS = ["P11", "P14", "P15", "P10", "P13"]
Y_OBSERVED = np.asarray([1, 1, 1, 0, 0], dtype=int)
STATUS = [
    "Tumour-positive node",
    "Tumour-positive node",
    "Tumour-positive node",
    "Histologically tumour-negative sampled node",
    "Histologically tumour-negative sampled node",
]
HISTOLOGY = ["SCC", "LUAD", "SCC", "SCC", "LUAD"]
CLASS_ORDER = [
    "Cer", "dhCer", "MHC", "DHC", "THC", "GM3", "SM", "PC", "PC(O)", "PC(P)",
    "LPC", "LPC(O)", "PE", "PE(O)", "PE(P)", "LPE", "PI", "PS", "PG", "CE",
    "Carnitine", "OxPC", "DG", "TG", "FFA",
]


def load_lipid_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    lipids: list[str] = []
    values: list[list[float]] = []
    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        lipids.append(row[0].strip())
        try:
            values.append([float(value) for value in row[1:]])
        except ValueError as exc:
            raise ValueError(f"Non-numeric abundance in {path.name}: {row}") from exc
    return lipids, np.asarray(values, dtype=float)


def infer_lipid_class(label: str) -> str:
    label = label.strip()
    if label.startswith("LPC (O-"):
        return "LPC(O)"
    if label.startswith("PC(O-"):
        return "PC(O)"
    if label.startswith("PC(P-"):
        return "PC(P)"
    if label.startswith("PE(O-"):
        return "PE(O)"
    if label.startswith("PE(P-"):
        return "PE(P)"
    if re.match(r"^C(?:14|16|18|20)(?:-| )", label):
        return "Carnitine"
    if label.startswith("GM3"):
        return "GM3"
    return label.split()[0]


def exact_assignments(n: int = 5, n_positive: int = 3) -> list[np.ndarray]:
    assignments: list[np.ndarray] = []
    for positive_indices in itertools.combinations(range(n), n_positive):
        labels = np.zeros(n, dtype=int)
        labels[list(positive_indices)] = 1
        assignments.append(labels)
    return assignments


def fit_lopo_ridge(X: np.ndarray, y: np.ndarray, C: float) -> dict[str, object]:
    probabilities: list[float] = []
    predictions: list[int] = []
    coefficients: list[np.ndarray] = []
    for held_out in range(len(y)):
        train_mask = np.arange(len(y)) != held_out
        scaler = StandardScaler().fit(X[train_mask])
        X_train = scaler.transform(X[train_mask])
        X_test = scaler.transform(X[[held_out]])
        model = LogisticRegression(
            C=C,
            l1_ratio=0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=10000,
            random_state=0,
        )
        model.fit(X_train, y[train_mask])
        probability = float(model.predict_proba(X_test)[0, 1])
        probabilities.append(probability)
        predictions.append(int(probability >= 0.5))
        coefficients.append(model.coef_[0].copy())

    probability_array = np.asarray(probabilities, dtype=float)
    prediction_array = np.asarray(predictions, dtype=int)
    return {
        "accuracy": float(accuracy_score(y, prediction_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction_array)),
        "roc_auc": float(roc_auc_score(y, probability_array)),
        "brier": float(brier_score_loss(y, probability_array)),
        "probabilities": probability_array,
        "predictions": prediction_array,
        "coefficients": np.vstack(coefficients),
    }


def exact_p_value(observed: float, null_values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(null_values) >= observed - 1e-12))


def permanova_pseudo_f(X: np.ndarray, labels: np.ndarray) -> float:
    """Two-group Euclidean pseudo-F after label-independent standardization."""
    Z = StandardScaler().fit_transform(X)
    distances_squared = squareform(pdist(Z, metric="euclidean")) ** 2
    n = len(labels)
    groups = np.unique(labels)
    ss_total = np.sum(np.triu(distances_squared, 1)) / n
    ss_within = 0.0
    for group in groups:
        idx = np.where(labels == group)[0]
        if len(idx) > 1:
            ss_within += np.sum(np.triu(distances_squared[np.ix_(idx, idx)], 1)) / len(idx)
    ss_between = ss_total - ss_within
    df_between = len(groups) - 1
    df_within = n - len(groups)
    if ss_within <= 0:
        return float("inf")
    return float((ss_between / df_between) / (ss_within / df_within))


def write_csv(path: Path, header: Sequence[object], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def panel_a_probabilities(
    class_delta: np.ndarray,
    y: np.ndarray,
    out_path: Path,
) -> dict[str, object]:
    primary = fit_lopo_ridge(class_delta, y, PRIMARY_C)
    sensitivity_cs = [0.03, 0.1, 0.3, 1.0]
    all_probabilities = np.vstack([
        np.asarray(fit_lopo_ridge(class_delta, y, C)["probabilities"]) for C in sensitivity_cs
    ])
    central = np.asarray(primary["probabilities"])
    lower = central - all_probabilities.min(axis=0)
    upper = all_probabilities.max(axis=0) - central

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    positions = np.arange(len(PATIENTS))
    positive_idx = np.where(y == 1)[0]
    negative_idx = np.where(y == 0)[0]
    ax.bar(positive_idx, central[positive_idx],
           yerr=np.vstack([lower[positive_idx], upper[positive_idx]]), capsize=5,
           label="Tumour-positive node")
    ax.bar(negative_idx, central[negative_idx],
           yerr=np.vstack([lower[negative_idx], upper[negative_idx]]), capsize=5,
           label="Histologically tumour-negative sampled node")
    ax.axhline(0.5, linestyle="--", linewidth=1.2, label="Decision threshold")
    for position, value in zip(positions, central):
        ax.text(position, value + 0.035, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{p}\n{h}" for p, h in zip(PATIENTS, HISTOLOGY)])
    ax.set_ylabel("LOPO probability of a tumour-positive node")
    ax.set_xlabel("Patient")
    ax.set_title("LOPO probabilities from paired lipid-class shifts")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return primary


def panel_b_permutation(
    class_delta: np.ndarray,
    y: np.ndarray,
    assignments: list[np.ndarray],
    out_path: Path,
) -> tuple[list[dict[str, object]], float]:
    results = [fit_lopo_ridge(class_delta, labels, PRIMARY_C) for labels in assignments]
    balanced = [float(result["balanced_accuracy"]) for result in results]
    observed_index = next(i for i, labels in enumerate(assignments) if np.array_equal(labels, y))
    observed = balanced[observed_index]
    p_exact = exact_p_value(observed, balanced)

    labels_text = ["".join(str(int(v)) for v in labels) for labels in assignments]
    positions = np.arange(len(assignments))
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    other_idx = np.asarray([i for i in range(len(assignments)) if i != observed_index])
    ax.bar(other_idx, np.asarray(balanced)[other_idx], label="Alternative exact assignments")
    ax.bar([observed_index], [observed], label="Observed nodal-status assignment")
    ax.scatter([observed_index], [observed], marker="*", s=220)
    for position, value in zip(positions, balanced):
        ax.text(position, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(observed, linestyle="--", linewidth=1.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels_text, rotation=45, ha="right")
    ax.set_ylabel("LOPO balanced accuracy")
    ax.set_xlabel("All exact 3-positive/2-negative label assignments\n(order: P11, P14, P15, P10, P13)")
    ax.set_title(f"Exact permutation calibration gives P = {p_exact:.2f}")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return results, p_exact


def panel_c_coefficients(
    class_delta: np.ndarray,
    y: np.ndarray,
    out_path: Path,
) -> list[list[object]]:
    primary = fit_lopo_ridge(class_delta, y, PRIMARY_C)
    coefficient_matrix = np.asarray(primary["coefficients"])
    means = coefficient_matrix.mean(axis=0)
    sds = coefficient_matrix.std(axis=0, ddof=1)
    order = np.argsort(np.abs(means))[-10:]
    order = order[np.argsort(means[order])]

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ypos = np.arange(len(order))
    selected_means = means[order]
    selected_sds = sds[order]
    negative_positions = np.where(selected_means < 0)[0]
    positive_positions = np.where(selected_means >= 0)[0]
    ax.barh(negative_positions, selected_means[negative_positions],
            xerr=selected_sds[negative_positions], capsize=4, label="Toward tumour-negative status")
    ax.barh(positive_positions, selected_means[positive_positions],
            xerr=selected_sds[positive_positions], capsize=4, label="Toward tumour-positive status")
    ax.axvline(0, linewidth=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([CLASS_ORDER[i] for i in order])
    ax.set_xlabel("Mean standardized ridge coefficient across five LOPO fits")
    ax.set_title("Lipid-class coefficients across five LOPO training folds")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows: list[list[object]] = []
    for idx in np.argsort(-np.abs(means)):
        rows.append([CLASS_ORDER[idx], float(means[idx]), float(sds[idx]), int(np.sum(coefficient_matrix[:, idx] > 0)),
                     int(np.sum(coefficient_matrix[:, idx] < 0))])
    return rows


def combine_panels(panel_paths: Sequence[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in panel_paths]
    max_width = max(image.width for image in images)
    resized: list[Image.Image] = []
    for image in images:
        if image.width != max_width:
            height = int(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        resized.append(image)
    margin = 60
    label_height = 80
    canvas = Image.new("RGB", (max_width + 2 * margin, sum(im.height + label_height for im in resized) + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 58)
    except OSError:
        font = ImageFont.load_default()
    y = margin
    for label, image in zip(["A", "B", "C"], resized):
        draw.text((margin, y), label, fill="black", font=font)
        y += label_height
        canvas.paste(image, (margin, y))
        y += image.height
    canvas.save(output_path, dpi=(300, 300))


def make_sensitivity_figure(rows: list[list[object]], out_path: Path) -> None:
    # Bars at the prespecified C=0.1, comparing feature representations.
    primary_rows = [row for row in rows if math.isclose(float(row[2]), PRIMARY_C)]
    labels = [str(row[0]).replace(", ", "\n") for row in primary_rows]
    bacc = [float(row[4]) for row in primary_rows]
    auc = [float(row[5]) for row in primary_rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.bar(x - width / 2, bacc, width, label="Balanced accuracy")
    ax.bar(x + width / 2, auc, width, label="ROC AUC")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("LOPO performance estimate")
    ax.set_title("Apparent performance changes with the feature representation")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_regularization_figure(rows: list[list[object]], out_path: Path) -> None:
    class_rows = [row for row in rows if row[0] == "25 prespecified lipid-class totals"]
    x = np.arange(len(class_rows))
    width = 0.28
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.bar(x - width, [float(r[3]) for r in class_rows], width, label="Accuracy")
    ax.bar(x, [float(r[4]) for r in class_rows], width, label="Balanced accuracy")
    ax.bar(x + width, [float(r[5]) for r in class_rows], width, label="ROC AUC")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r[2]) for r in class_rows])
    ax.set_xlabel("Inverse regularization strength C")
    ax.set_ylabel("LOPO performance estimate")
    ax.set_title("Performance estimates are sensitive to regularization")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(positive_csv: Path, negative_csv: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    lipids, positive = load_lipid_csv(positive_csv)
    negative_lipids, negative = load_lipid_csv(negative_csv)
    if lipids != negative_lipids:
        raise ValueError("Positive- and negative-node files do not contain the same ordered lipid list.")

    primary = np.vstack([positive[:, 0], positive[:, 1], positive[:, 2], negative[:, 0], negative[:, 1]])
    node = np.vstack([positive[:, 3], positive[:, 4], positive[:, 5], negative[:, 2], negative[:, 3]])
    if np.any(primary <= 0) or np.any(node <= 0):
        raise ValueError("All abundances must be positive for log2(Node/Primary).")
    delta = np.log2(node / primary)

    classes = np.asarray([infer_lipid_class(label) for label in lipids])
    unexpected = sorted(set(classes) - set(CLASS_ORDER))
    if unexpected:
        raise ValueError(f"Unrecognized lipid classes: {unexpected}")

    class_delta = np.zeros((len(PATIENTS), len(CLASS_ORDER)), dtype=float)
    for class_index, lipid_class in enumerate(CLASS_ORDER):
        idx = classes == lipid_class
        if not np.any(idx):
            raise ValueError(f"No species found for prespecified class {lipid_class}")
        class_delta[:, class_index] = np.log2(node[:, idx].sum(axis=1) / primary[:, idx].sum(axis=1))

    floor_mask = (primary <= 1e-6) | (node <= 1e-6)
    floor_features = floor_mask.any(axis=0)
    nonfloor_delta = delta[:, ~floor_features]

    assignments = exact_assignments()
    representations = [
        ("306 species, all values", delta),
        (f"{nonfloor_delta.shape[1]} species, numerical-floor features excluded", nonfloor_delta),
        ("25 prespecified lipid-class totals", class_delta),
    ]

    model_rows: list[list[object]] = []
    prediction_rows: list[list[object]] = []
    permutation_rows: list[list[object]] = []
    for representation_name, X in representations:
        for C in C_GRID:
            observed = fit_lopo_ridge(X, Y_OBSERVED, C)
            permuted = [fit_lopo_ridge(X, labels, C) for labels in assignments]
            p_accuracy = exact_p_value(float(observed["accuracy"]), [float(r["accuracy"]) for r in permuted])
            p_balanced = exact_p_value(float(observed["balanced_accuracy"]), [float(r["balanced_accuracy"]) for r in permuted])
            p_auc = exact_p_value(float(observed["roc_auc"]), [float(r["roc_auc"]) for r in permuted])
            model_rows.append([
                representation_name, X.shape[1], C, observed["accuracy"], observed["balanced_accuracy"],
                observed["roc_auc"], observed["brier"], p_accuracy, p_balanced, p_auc,
            ])
            for patient_index, patient in enumerate(PATIENTS):
                prediction_rows.append([
                    representation_name, C, patient, HISTOLOGY[patient_index], STATUS[patient_index],
                    int(Y_OBSERVED[patient_index]), float(np.asarray(observed["probabilities"])[patient_index]),
                    int(np.asarray(observed["predictions"])[patient_index]),
                    int(np.asarray(observed["predictions"])[patient_index] == Y_OBSERVED[patient_index]),
                ])
            for permutation_index, (labels, result) in enumerate(zip(assignments, permuted), start=1):
                permutation_rows.append([
                    representation_name, C, permutation_index, "".join(str(int(v)) for v in labels),
                    result["accuracy"], result["balanced_accuracy"], result["roc_auc"], result["brier"],
                    int(np.array_equal(labels, Y_OBSERVED)),
                ])

    permanova_rows: list[list[object]] = []
    for representation_name, X in representations:
        observed_f = permanova_pseudo_f(X, Y_OBSERVED)
        null_f = [permanova_pseudo_f(X, labels) for labels in assignments]
        permanova_rows.append([
            representation_name, X.shape[1], observed_f, exact_p_value(observed_f, null_f), min(null_f), max(null_f)
        ])

    # Main figure panels.
    primary_result = panel_a_probabilities(class_delta, Y_OBSERVED, figures_dir / "Figure_4A_LOPO_probabilities.png")
    permutation_results, p_exact = panel_b_permutation(
        class_delta, Y_OBSERVED, assignments, figures_dir / "Figure_4B_exact_permutation.png"
    )
    coefficient_rows = panel_c_coefficients(
        class_delta, Y_OBSERVED, figures_dir / "Figure_4C_lipid_class_contributions.png"
    )
    combine_panels(
        [figures_dir / "Figure_4A_LOPO_probabilities.png",
         figures_dir / "Figure_4B_exact_permutation.png",
         figures_dir / "Figure_4C_lipid_class_contributions.png"],
        figures_dir / "Figure_4_permutation_calibrated_learning.png",
    )
    make_sensitivity_figure(model_rows, figures_dir / "Figure_S3_feature_representation_sensitivity.png")
    make_regularization_figure(model_rows, figures_dir / "Figure_S4_regularization_sensitivity.png")

    # Main tables.
    write_csv(
        tables_dir / "paired_delta_306_species.csv",
        ["Patient", "Histology", "Nodal_status"] + lipids,
        [[PATIENTS[i], HISTOLOGY[i], STATUS[i]] + delta[i].tolist() for i in range(5)],
    )
    write_csv(
        tables_dir / "paired_delta_25_lipid_classes.csv",
        ["Patient", "Histology", "Nodal_status"] + CLASS_ORDER,
        [[PATIENTS[i], HISTOLOGY[i], STATUS[i]] + class_delta[i].tolist() for i in range(5)],
    )
    write_csv(
        tables_dir / "ML_model_sensitivity.csv",
        ["Feature_representation", "Number_of_features", "C", "Accuracy", "Balanced_accuracy", "ROC_AUC",
         "Brier_score", "Exact_P_accuracy", "Exact_P_balanced_accuracy", "Exact_P_ROC_AUC"],
        model_rows,
    )
    write_csv(
        tables_dir / "ML_patient_predictions.csv",
        ["Feature_representation", "C", "Patient", "Histology", "Nodal_status", "Observed_label",
         "Predicted_probability_positive", "Predicted_label", "Correct"],
        prediction_rows,
    )
    write_csv(
        tables_dir / "ML_exact_label_permutations.csv",
        ["Feature_representation", "C", "Permutation_number", "Label_pattern_P11_P14_P15_P10_P13",
         "Accuracy", "Balanced_accuracy", "ROC_AUC", "Brier_score", "Observed_assignment"],
        permutation_rows,
    )
    write_csv(
        tables_dir / "ML_exact_multivariate_permutation.csv",
        ["Feature_representation", "Number_of_features", "Observed_pseudo_F", "Exact_P", "Minimum_null_F", "Maximum_null_F"],
        permanova_rows,
    )
    write_csv(
        tables_dir / "ML_lipid_class_coefficients_LOPO.csv",
        ["Lipid_class", "Mean_standardized_coefficient", "SD_across_LOPO_fits", "Positive_sign_folds", "Negative_sign_folds"],
        coefficient_rows,
    )
    write_csv(
        tables_dir / "numerical_floor_affected_lipids.csv",
        ["Lipid", "Class", "Affected_patient_measurements"],
        [[lipids[j], classes[j], int(floor_mask[:, j].sum())] for j in np.where(floor_features)[0]],
    )

    # A concise deidentified clinical annotation table used only for interpretation, never model fitting.
    clinical_rows = [
        ["P10", 65, "SCC", "Tumour-negative sampled node", 26.6, "Former", "50–60%", "T4N2M0", "Station 7; primary specimen described as contiguous with/including mediastinal nodal tissue"],
        ["P11", 63, "SCC", "Tumour-positive node", 26.4, "Former", "5–10%", "T3N2M0", "Station 4L"],
        ["P13", 67, "LUAD", "Tumour-negative sampled node", 20.7, "Former", "<1%", "T4N0M0", "Station 4R; KRAS Q61L, STK11 Y261*, TP53 variants"],
        ["P14", 61, "LUAD", "Tumour-positive node", 30.4, "Current", "5%", "T3N2M0", "Station 4R; KRAS G13D, TP53 S241F"],
        ["P15", 58, "SCC", "Tumour-positive node", 34.9, "Current", "<1%", "T2N3M0", "Station 7; pathology-positive classification, internal vial-label anomaly requires reconciliation"],
    ]
    write_csv(
        tables_dir / "clinical_metadata_deidentified.csv",
        ["Patient", "Age", "Histology", "Nodal_status", "BMI", "Smoking_status", "Reported_PD_L1", "Recorded_TNM", "Key_annotation"],
        clinical_rows,
    )

    observed_index = next(i for i, labels in enumerate(assignments) if np.array_equal(labels, Y_OBSERVED))
    summary = {
        "primary_model": {
            "feature_representation": "25 prespecified lipid-class paired shifts",
            "algorithm": "L2-penalized logistic regression",
            "validation": "leave-one-patient-out cross-validation",
            "C": PRIMARY_C,
            "n_patients": 5,
            "n_features": 25,
            "accuracy": primary_result["accuracy"],
            "balanced_accuracy": primary_result["balanced_accuracy"],
            "roc_auc": primary_result["roc_auc"],
            "brier_score": primary_result["brier"],
            "exact_permutation_P_balanced_accuracy": p_exact,
            "observed_assignment_index": observed_index + 1,
        },
        "cohort": {
            "patients": PATIENTS,
            "tumour_positive": 3,
            "tumour_negative_sampled_nodes": 2,
            "lipid_species": len(lipids),
            "lipid_classes": len(CLASS_ORDER),
            "floor_affected_species": int(floor_features.sum()),
            "nonfloor_species": int((~floor_features).sum()),
        },
        "interpretation": (
            "The analysis is exploratory. Apparent LOPO discrimination was not exceptional under exhaustive exact "
            "label permutation, varied with regularization and feature representation, and cannot be treated as "
            "a validated classifier or biomarker panel."
        ),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = f"""NSCLC paired lipidomics permutation-calibrated exploratory learning\n\nPrimary input\n- {positive_csv.name}\n- {negative_csv.name}\n\nBiological unit\n- Five NSCLC patients: P11/P14/P15 tumour-positive nodes; P10/P13 histologically tumour-negative sampled nodes.\n- Patient 12 is not used in supervised learning because it is a non-malignant disease comparator.\n\nPrimary analysis\n- Within-patient delta = log2(Node/Primary).\n- 25 prespecified lipid-class totals.\n- L2-penalized logistic regression, C={PRIMARY_C}.\n- Scaling inside each leave-one-patient-out training fold.\n- Exact enumeration of all 10 possible three-positive/two-negative label assignments.\n\nPrimary result\n- Accuracy: {float(primary_result['accuracy']):.3f}\n- Balanced accuracy: {float(primary_result['balanced_accuracy']):.3f}\n- ROC AUC: {float(primary_result['roc_auc']):.3f}\n- Exact P for balanced accuracy: {p_exact:.2f}\n\nInterpretation\nThis is a stress test of recoverable structure in a five-patient pilot cohort, not predictive validation. Clinical variables are retained as descriptive annotations and potential confounders; they are not added as model predictors because doing so would be statistically indefensible at n=5.\n"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive", type=Path, required=True, help="Tumour-positive paired lipid CSV")
    parser.add_argument("--negative", type=Path, required=True, help="Tumour-negative paired lipid CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(args.positive, args.negative, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
