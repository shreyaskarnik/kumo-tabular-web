"""M2: golden fixtures for the JS preprocessing (estimator 0).

Runs sdm's KumoTabular (small) with a deterministic copy of the default
recipe: the random augmentations (FlipSign, ShuffleColumns,
ShuffleCategories) are removed, everything else is unchanged. The browser
uses the same deterministic recipe, so every value can be compared exactly.

Per dataset it writes fixtures/<name>/:
  data.csv       raw rows, context first; query targets kept for scoring
  meta.json      target, task, context count, stypes, class order, shapes
  x.bin y.bin mask.bin   the inner model's inputs (float32 / int32|float32 / uint8)
  logits.bin     the inner model's output
  pred.bin       final predictions: class probabilities [Rq, K] or point estimates [Rq]
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

import sdm
import sdm.processing as sp

OUT = Path(__file__).parent.parent / "fixtures"


def deterministic_recipe() -> sp.Recipe:
    def numerical_processor() -> sp.Sequential:
        return sp.Sequential(
            sp.Cast(torch.float64),
            sp.DropConstantColumns(),
            sp.Standardize(eps=1e-6),
            sp.Clip(-100.0, 100.0),
            sp.Identity(),  # estimator 0 of the round-robin Choice
            sp.ClipSigma(threshold=4.0),
        )

    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                numerical=[numerical_processor()],
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.AddCategoryCounts(min_cardinality=50),
                    sp.ToNumerical(),
                    numerical_processor(),
                ],
            ),
            sp.SelectColumns(500, method="first"),
            sp.Cast(torch.float32),
        ],
        target=[
            sp.StypeDispatch(
                categorical=[sp.AlignCategories()],
                numerical=[sp.Standardize()],
            ),
        ],
        output=sp.TaskDispatch(
            classification=[sp.AverageEstimators(), sp.Softmax()],
            regression=[sp.SortQuantiles(), sp.AverageEstimators(trim_fraction=0.2)],
        ),
    )


def mixed(n: int, seed: int, task: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    city = rng.choice([f"city_{i:02d}" for i in range(80)], n)  # > 50 values: count column
    plan = rng.choice(["basic", "plus", "pro", None], n, p=[0.4, 0.3, 0.2, 0.1])
    tenure = rng.gamma(2.0, 12.0, n)
    tenure[rng.random(n) < 0.1] = np.nan
    spend = rng.lognormal(3.0, 1.0, n)
    spend[rng.random(n) < 0.02] *= 1000  # outliers for the clipping steps
    tickets = rng.poisson(2.0, n).astype(float)
    score = 0.03 * tenure.clip(0, 60) - 0.4 * tickets + (plan == "pro") * 1.5 + rng.normal(0, 1, n)
    score = np.nan_to_num(score)
    df = pd.DataFrame({"city": city, "plan": plan, "tenure": tenure, "spend": spend,
                       "tickets": tickets, "constant": 7.0, "active": rng.random(n) < 0.5})
    if task == "classification":
        df["churn"] = np.where(score < np.quantile(score, 0.35), "yes", "no")
    else:
        df["ltv"] = 100 + 40 * score + rng.normal(0, 5, n)
    return df


def datasets():
    bc = load_breast_cancer(as_frame=True).frame
    iris = load_iris(as_frame=True).frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    diab = load_diabetes(as_frame=True).frame
    return [
        ("breast-cancer", bc, "target", "classification", 400),
        ("iris", iris, "target", "classification", 100),
        ("diabetes", diab, "target", "regression", 300),
        ("mixed-churn", mixed(600, 0, "classification"), "churn", "classification", 450),
        ("mixed-ltv", mixed(500, 1, "regression"), "ltv", "regression", 380),
    ]


def main():
    models = {}
    for name, df, target, task, n_context in datasets():
        model = models.setdefault(task, sdm.models.KumoTabular(size="small", task=task, device="cpu").eval())
        inner = model.models[task]
        seen = {}

        def hook(_m, args, kwargs, output):
            seen.update(x=kwargs.get("x", args[0] if args else None).detach().clone(),
                        y=kwargs.get("y", args[1] if len(args) > 1 else None).detach().clone(),
                        mask=kwargs.get("categorical_mask", args[2] if len(args) > 2 else None).detach().clone(),
                        out=output.detach().clone())

        handle = inner.register_forward_hook(hook, with_kwargs=True)
        stypes = sdm.infer_stypes(df, overrides={target: "categorical" if task == "classification" else "numerical"})
        table = sdm.TableTensor.from_pandas(df=df, stypes=stypes, device="cpu")
        with torch.inference_mode():
            result = model(x_context=table[:n_context].drop_columns(target), y_context=table[:n_context, target],
                           x_query=table[n_context:].drop_columns(target), recipe=deterministic_recipe())
        handle.remove()

        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(folder / "data.csv", index=False)
        x, y, mask, logits = seen["x"], seen["y"], seen["mask"], seen["out"]
        while x.dim() > 2:
            x, y, mask, logits = x[0], y[0], mask[0], logits[0]
        x.float().numpy().tofile(folder / "x.bin")
        (y.numpy().astype(np.int32) if task == "classification" else y.float().numpy()).tofile(folder / "y.bin")
        mask.bool().numpy().astype(np.uint8).tofile(folder / "mask.bin")
        logits.float().numpy().tofile(folder / "logits.bin")

        out = result.numerical.float()
        while out.dim() > 2:
            out = out[0]
        columns = list(result.columns[sdm.Stype.numerical])
        if task == "classification":
            pred = out.numpy()  # [Rq, K] probabilities, columns = class labels
        else:
            pred = out.mean(dim=-1).numpy()  # point estimate: mean of sorted quantiles
        pred.astype(np.float32).tofile(folder / "pred.bin")

        meta = {
            "name": name, "task": task, "target": target, "context": n_context, "rows": len(df),
            "stypes": {k: str(v).split(".")[-1] for k, v in stypes.items()},
            "classes": columns if task == "classification" else None,
            "x": list(x.shape), "y": list(y.shape), "logits": list(logits.shape), "pred": list(pred.shape),
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=1))
        print(name, {k: meta[k] for k in ("x", "y", "logits", "pred", "classes")}, "mask", mask.int().tolist())


if __name__ == "__main__":
    main()
