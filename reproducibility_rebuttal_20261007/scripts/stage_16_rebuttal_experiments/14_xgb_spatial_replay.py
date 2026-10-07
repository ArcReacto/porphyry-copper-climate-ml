"""Replay fold-local spatial graphs with the XGBoost downstream classifier."""

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
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "data/run_inputs/known_mining_neutral_ratio_1_10_supervised_all_features_v1"
DEFAULT_GRAPH = ROOT / "outputs/rebuttal_experiments/02_spatial_block_validation"
DEFAULT_OUTPUT = ROOT / "outputs/rebuttal_experiments/xgboost/02_spatial_block_validation"
METRICS = ["roc_auc", "average_precision", "f1", "top05_f1", "top05_ndcg", "top10_f1", "top10_ndcg"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-degrees", nargs="+", type=float, default=[2.0, 3.0])
    parser.add_argument("--spearman-threshold", type=float, default=0.30)
    args = parser.parse_args()
    replay = load_module("rebuttal_graph_xgb_replay", SCRIPT_DIR / "13_xgb_graphunion_replay.py")
    spatial = load_module("rebuttal_spatial_rf_protocol", SCRIPT_DIR / "02_spatial_block_validation.py")
    base = replay.load_base()
    df, dataset_path = base.load_dataset_from_run(args.run_dir)
    df = df[df[base.TARGET].isin([0, 1])].reset_index(drop=True)
    y = df[base.TARGET].astype(int)
    fsets = base.feature_sets(pd.read_csv(args.run_dir / "00_dataset_profile/feature_roles.csv"))
    graph = pd.read_csv(args.graph_dir / "spatial_fold_graph_targets.csv")
    manifest = pd.read_csv(args.graph_dir / "spatial_fold_manifest.csv")
    source_predictions = pd.read_csv(
        args.graph_dir / "spatial_oof_predictions.csv",
        usecols=["grid_degrees", "fold", "model", "row_index"],
    )
    metrics, predictions = [], []

    for degrees in args.grid_degrees:
        block_ids = spatial.spatial_block_ids(df, degrees)
        for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(df, y, block_ids), start=1):
            source = manifest.loc[np.isclose(manifest["grid_degrees"], degrees) & manifest["fold"].eq(fold)]
            if len(source) != 1 or int(source.iloc[0]["n_train"]) != len(train_idx) or int(source.iloc[0]["n_test"]) != len(test_idx):
                raise ValueError(f"Frozen spatial graph does not match {degrees} degree fold {fold}")
            source_rows = set(source_predictions.loc[
                np.isclose(source_predictions["grid_degrees"], degrees)
                & source_predictions["fold"].eq(fold)
                & source_predictions["model"].eq("NoClimate_RF"),
                "row_index",
            ].astype(int))
            if source_rows != set(map(int, test_idx)):
                raise ValueError(f"Frozen spatial graph test rows do not match {degrees} degree fold {fold}")
            train, test = df.iloc[train_idx], df.iloc[test_idx]
            graph_cols = set(graph.loc[
                np.isclose(graph["grid_degrees"], degrees) & graph["fold"].eq(fold) & graph["residualizable"],
                "column",
            ])
            recipes = {}
            for key, internal in [
                ("M2_NoClimate_XGB", "NoClimate_RF"),
                ("M4_Spearman_XGB", "M4_Spearman_RF"),
                ("M4_GraphUnion_XGB", "M4_GraphUnion_RF"),
            ]:
                recipes[key], _ = base.make_recipe(internal, train, fsets, graph_cols, args.spearman_threshold)
            geo = fsets["geochemistry"]
            recipes["M3_Broad_XGB"] = base.FeatureRecipe(
                "M3_Broad_XGB", [c for c in fsets["no_climate"] if c not in set(geo)],
                geo, fsets["climate_adjusters"],
            ).fit(train)
            test_group = ";".join(sorted(block_ids.iloc[test_idx].unique()))
            for key, recipe in recipes.items():
                model = make_xgb_model(y.iloc[train_idx])
                model.fit(recipe.transform(train), y.iloc[train_idx])
                before_metrics, before_preds = len(metrics), len(predictions)
                base.add_metric_row(
                    metrics, predictions, dataset=args.run_dir.name,
                    cv_name="spatial_block_kfold", fold=fold, test_group=test_group,
                    model_key=key, used_cols=recipe.output_columns(),
                    train_idx=np.asarray(train_idx), test_idx=np.asarray(test_idx),
                    df=df, y_prob=model.predict_proba(recipe.transform(test))[:, 1],
                )
                metrics[before_metrics]["grid_degrees"] = degrees
                for row in predictions[before_preds:]:
                    row["grid_degrees"] = degrees
                    row["spatial_block"] = block_ids.iloc[row["row_index"]]
                print(f"grid={degrees:g} fold={fold} model={key} AP={metrics[-1]['average_precision']:.4f}", flush=True)

    metric_df = pd.DataFrame(metrics)
    paired = []
    for degrees, current in metric_df.groupby("grid_degrees"):
        base_rows = current[current["model"].eq("M2_NoClimate_XGB")].set_index("fold")
        for key in current["model"].unique():
            if key == "M2_NoClimate_XGB":
                continue
            candidate = current[current["model"].eq(key)].set_index("fold")
            for metric in METRICS:
                delta = candidate[metric] - base_rows[metric]
                paired.append({"grid_degrees": degrees, "model": key, "metric": metric,
                               "mean_delta": delta.mean(), "better_folds": int((delta > 0).sum()),
                               "fold_deltas": json.dumps(delta.round(6).tolist())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(args.output_dir / "spatial_fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output_dir / "spatial_oof_predictions.csv", index=False)
    metric_df.groupby(["grid_degrees", "model"])[METRICS].agg(["mean", "std"]).to_csv(args.output_dir / "spatial_model_summary.csv")
    pd.DataFrame(paired).to_csv(args.output_dir / "spatial_paired_comparison.csv", index=False)
    run_manifest = {"input_dataset": str(dataset_path), "source_graph_dir": str(args.graph_dir),
                    "classifier": "XGBoost", "xgboost_parameters": PARAMETERS,
                    "positive_weight": "computed within training fold",
                    "graph_selection": "frozen, spatial-fold-local targets from experiment 02; RF-based stability selector unchanged",
                    "split_check": "same grid, fold, train/test counts and OOF test row indices as the frozen graph run"}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"Wrote XGBoost spatial results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
