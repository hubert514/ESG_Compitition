from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_FIELDS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

DEFAULT_LABEL_ORDERS = {
    "promise_status": ["Yes", "No", "N/A"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "more_than_5_years",
        "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}

TRUE_KEY_CANDIDATES = [
    "{field}",
    "true_{field}",
    "{field}_true",
    "label_{field}",
    "{field}_label",
    "gold_{field}",
    "{field}_gold",
    "gt_{field}",
    "{field}_gt",
]

PRED_KEY_CANDIDATES = [
    "{field}_pred",
    "pred_{field}",
    "{field}_prediction",
    "prediction_{field}",
    "{field}_predicted",
    "predicted_{field}",
    "{field}_output",
    "output_{field}",
]

ROOT_LIST_KEYS = ["predictions", "results", "items", "data"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot confusion matrices from a prediction JSON file."
    )
    parser.add_argument("input_json", type=Path, help="Path to the prediction JSON file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path. Default: next to input JSON.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Target fields to plot.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(16, 12),
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output figure DPI.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ROOT_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            raise ValueError(
                "JSON root is a dict, but no record list was found under "
                f"{ROOT_LIST_KEYS}."
            )
    else:
        raise ValueError("Unsupported JSON format. Expected a list or a dict.")

    if not records or not isinstance(records[0], dict):
        raise ValueError("Prediction JSON does not contain a list of object records.")
    return records


def find_key(record: dict[str, Any], field: str, candidates: list[str]) -> str | None:
    for template in candidates:
        key = template.format(field=field)
        if key in record:
            return key
    return None


def get_nested_prediction(record: dict[str, Any], field: str) -> Any:
    for container_key in ["predictions", "prediction", "pred", "outputs"]:
        container = record.get(container_key)
        if isinstance(container, dict) and field in container:
            return container[field]
    return None


def get_nested_truth(record: dict[str, Any], field: str) -> Any:
    for container_key in ["labels", "label", "ground_truth", "truth", "targets"]:
        container = record.get(container_key)
        if isinstance(container, dict) and field in container:
            return container[field]
    return None


def normalize_value(value: Any) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    return text if text else "N/A"


def build_confusion_matrix(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> list[list[int]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        if true_label not in label_to_index or pred_label not in label_to_index:
            continue
        matrix[label_to_index[true_label]][label_to_index[pred_label]] += 1
    return matrix


def compute_macro_f1(matrix: list[list[int]]) -> float:
    f1_scores: list[float] = []
    size = len(matrix)
    for class_index in range(size):
        tp = matrix[class_index][class_index]
        fp = sum(matrix[row][class_index] for row in range(size) if row != class_index)
        fn = sum(matrix[class_index][col] for col in range(size) if col != class_index)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def collect_field_pairs(
    records: list[dict[str, Any]], field: str
) -> tuple[list[str], list[str], str, str]:
    sample = records[0]
    true_key = find_key(sample, field, TRUE_KEY_CANDIDATES)
    pred_key = find_key(sample, field, PRED_KEY_CANDIDATES)

    y_true: list[str] = []
    y_pred: list[str] = []

    for record in records:
        true_value = record.get(true_key) if true_key else get_nested_truth(record, field)
        pred_value = record.get(pred_key) if pred_key else get_nested_prediction(record, field)
        if true_value is None or pred_value is None:
            continue
        y_true.append(normalize_value(true_value))
        y_pred.append(normalize_value(pred_value))

    if not y_true or not y_pred:
        available_keys = sorted(sample.keys())
        raise ValueError(
            f"Could not find usable true/prediction values for '{field}'. "
            f"Top-level keys in the first record: {available_keys}"
        )

    return y_true, y_pred, true_key or "<nested>", pred_key or "<nested>"


def resolve_labels(field: str, y_true: list[str], y_pred: list[str]) -> list[str]:
    observed = list(dict.fromkeys([*y_true, *y_pred]))
    default_order = DEFAULT_LABEL_ORDERS.get(field)
    if default_order is None:
        return observed

    labels = [label for label in default_order if label in observed]
    extra = [label for label in observed if label not in labels]
    return labels + extra


def plot_field(ax: plt.Axes, field: str, y_true: list[str], y_pred: list[str]) -> None:
    labels = resolve_labels(field, y_true, y_pred)
    matrix = build_confusion_matrix(y_true, y_pred, labels)
    macro_f1 = compute_macro_f1(matrix)

    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    flat_values = [value for row in matrix for value in row]
    threshold = max(flat_values) / 2 if flat_values else 0
    for row_index in range(len(matrix)):
        for col_index in range(len(matrix[row_index])):
            value = matrix[row_index][col_index]
            color = "white" if value > threshold else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:d}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    ax.set_title(f"{field} (Macro F1: {macro_f1:.4f})", fontsize=11, pad=10, weight="bold")
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)


def main() -> None:
    args = parse_args()
    input_path = args.input_json.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_confusion_matrices.png")
    )

    records = load_records(input_path)

    fig, axes = plt.subplots(2, 2, figsize=tuple(args.figsize))
    axes_flat = axes.flatten()

    plotted_fields: list[str] = []
    for ax, field in zip(axes_flat, args.fields):
        y_true, y_pred, true_key, pred_key = collect_field_pairs(records, field)
        plot_field(ax, field, y_true, y_pred)
        plotted_fields.append(field)
        print(f"{field}: true='{true_key}', pred='{pred_key}', samples={len(y_true)}")

    for ax in axes_flat[len(plotted_fields) :]:
        ax.axis("off")

    fig.suptitle("Ensemble Prediction Confusion Matrices", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
