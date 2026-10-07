"""Evaluate frozen fold-local graph targets with the headline XGBoost learner.

The graph targets were constructed without test-fold data by experiment 01.
This script changes only the downstream classifier, not the graph algorithm.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from xgb_model import PARAMETERS, make_xgb_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "data/run_inputs/known_mining_neutral_ratio_1_10_supervised_all_features_v1"
DEFAULT_GRAPH = ROOT / "outputs/rebuttal_experiments/01_fold_local_graphunion"
DEFAULT_OUTPUT = ROOT / "outputs/rebuttal_experiments/xgboost/01_fold_local_graphunion"
METRICS = ["roc_auc", "average_precision", "f1", "top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"]


def load_base():
    path = ROOT / "scripts/stage_11_paper_optimization/63_full_feature_graph_guided_m4_and_perturbation.py"
    spec = importlib.util.spec_from_file_location("rebuttal_xgb_base63", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    args = parser.parse_args()
    base = load_base()
    df, dataset_path = base.load_dataset_from_run(args.run_dir)
    df = df[df[base.TARGET].isin([0, 1])].reset_index(drop=True)
    y = df[base.TARGET].astype(int)
    groups = df["state"].fillna("unknown").astype(str)
    role_table = pd.read_csv(args.run_dir / "00_dataset_profile/feature_roles.csv")
    fsets = base.feature_sets(role_table)
    graph = pd.read_csv(args.graph_dir / "fold_graph_raw_feature_targets.csv")
    fold_manifest = pd.read_csv(args.graph_dir / "fold_graph_manifest.csv").set_index("fold")
    source_predictions = pd.read_csv(args.graph_dir / "oof_predictions.csv", usecols=["fold", "model", "row_index"])
    metric_rows, prediction_rows, selection_rows = [], [], []

    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(df, y, groups), start=1):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        test_group = ";".join(sorted(groups.iloc[test_idx].unique()))
        source = fold_manifest.loc[fold]
        if (source["test_group"] != test_group or int(source["n_train"]) != len(train_idx)
                or int(source["n_test"]) != len(test_idx)):
            raise ValueError(f"Frozen graph fold {fold} does not match the current split")
        source_rows = set(source_predictions.loc[
            source_predictions["fold"].eq(fold) & source_predictions["model"].eq("NoClimate_RF"),
            "row_index",
        ].astype(int))
        if source_rows != set(map(int, test_idx)):
            raise ValueError(f"Frozen graph fold {fold} test rows do not match")
        graph_cols = set(graph.loc[(graph["fold"] == fold) & graph["residualizable"], "column"])
        recipes = {}
        for key, internal in [
            ("M1_Full_XGB", "Full_RF"),
            ("M2_NoClimate_XGB", "NoClimate_RF"),
            ("M4_Spearman_XGB", "M4_Spearman_RF"),
            ("M4_GraphUnion_XGB", "M4_GraphUnion_RF"),
        ]:
            recipe, selection = base.make_recipe(
                internal, train, fsets, graph_cols, args.spearman_threshold
            )
            recipes[key] = recipe
            if not selection.empty:
                selected = selection.loc[selection["selected_by_model"]].copy()
                selected.insert(0, "fold", fold)
                selected["model"] = key
                selection_rows.append(selected)
        geo = fsets["geochemistry"]
        nonclimate = fsets["no_climate"]
        climate = fsets["climate_adjusters"]
        recipes["M3_Broad_XGB"] = base.FeatureRecipe(
            "M3_Broad_XGB", [c for c in nonclimate if c not in set(geo)], geo, climate
        ).fit(train)

        for key, recipe in recipes.items():
            model = make_xgb_model(y.iloc[train_idx])
            model.fit(recipe.transform(train), y.iloc[train_idx])
            prob = model.predict_proba(recipe.transform(test))[:, 1]
            base.add_metric_row(
                metric_rows, prediction_rows, dataset=args.run_dir.name,
                cv_name="groupkfold_state", fold=fold, test_group=test_group,
                model_key=key, used_cols=recipe.output_columns(),
                train_idx=np.asarray(train_idx), test_idx=np.asarray(test_idx),
                df=df, y_prob=prob,
            )
            print(f"fold={fold} model={key} AP={metric_rows[-1]['average_precision']:.4f}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    baseline = metrics[metrics["model"].eq("M2_NoClimate_XGB")].set_index("fold")
    comparisons = []
    for key in metrics["model"].unique():
        if key == "M2_NoClimate_XGB":
            continue
        candidate = metrics[metrics["model"].eq(key)].set_index("fold")
        for metric in METRICS:
            delta = candidate[metric] - baseline[metric]
            comparisons.append({"model": key, "baseline": "M2_NoClimate_XGB", "metric": metric,
                                "mean_delta": float(delta.mean()), "better_folds": int((delta > 0).sum()),
                                "fold_deltas": json.dumps(delta.round(6).tolist())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.concat(selection_rows, ignore_index=True).to_csv(args.output_dir / "fold_selected_features.csv", index=False)
    metrics.groupby("model")[METRICS].agg(["mean", "std"]).to_csv(args.output_dir / "model_summary.csv")
    pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_comparison.csv", index=False)
    manifest = {"source_graph_dir": str(args.graph_dir), "input_dataset": str(dataset_path),
                "classifier": "XGBoost", "xgboost_parameters": PARAMETERS,
                "positive_weight": "computed from each training fold",
                "graph_selection": "frozen fold-local targets from experiment 01; RF-based stability selector unchanged",
                "validation": "five state GroupKFold splits checked against frozen graph manifest and OOF row indices"}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote XGBoost graph comparison to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
