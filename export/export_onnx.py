"""M1: export Kumo Tabular (small) classifier and regressor heads to ONNX.

The graph is the inner model `_KumoTabular.forward(x, y, categorical_mask)`:
  x                [R, C]  float32, context rows first, then query rows
  y_context        [Rc]    int64 class codes (classifier) / float32 (regressor)
  categorical_mask [C]     bool
  -> logits        [R - Rc, 10] (classifier) / [R - Rc, 999] quantiles (regressor)

Sample inputs come from a real `sdm` run (forward hook on the inner module),
so shapes and dtypes are exactly what the library feeds. Each head is checked
against PyTorch with ONNX Runtime (CPU) at three shapes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnxscript import opset20 as op
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, load_iris, load_wine

import sdm

HERE = Path(__file__).parent


def rem_scalar(a, b):
    # aten.remainder.Scalar with a symbolic divisor (index % num_columns).
    return op.Mod(a, op.CastLike(b, a), fmod=0)


class Head(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, y_context, categorical_mask):
        return self.inner(x, y_context, categorical_mask)


def capture(model, task, df, target, n_context):
    """Run sdm once and return the inner module's (x, y, mask) and output."""
    inner = model.models[task]
    seen = {}

    def hook(_module, args, kwargs, output):
        x = kwargs.get("x", args[0] if args else None)
        y = kwargs.get("y", args[1] if len(args) > 1 else None)
        mask = kwargs.get("categorical_mask", args[2] if len(args) > 2 else None)
        seen.update(x=x.detach().clone(), y=y.detach().clone(), mask=mask.detach().clone(), out=output.detach().clone())

    handle = inner.register_forward_hook(hook, with_kwargs=True)
    stypes = sdm.infer_stypes(df, overrides={target: "categorical" if task == "classification" else "numerical"})
    table = sdm.TableTensor.from_pandas(df=df, stypes=stypes, device="cpu")
    with torch.inference_mode():
        model(x_context=table[:n_context].drop_columns(target), y_context=table[:n_context, target],
              x_query=table[n_context:].drop_columns(target))
    handle.remove()
    # Drop leading ensemble/batch dims of size 1.
    x, y, mask = seen["x"], seen["y"], seen["mask"]
    while x.dim() > 2:
        x, y, mask, seen["out"] = x[0], y[0], mask[0], seen["out"][0]
    return x.float(), y, mask.bool(), seen["out"]


def datasets(task):
    if task == "classification":
        bc = load_breast_cancer(as_frame=True).frame
        iris = load_iris(as_frame=True).frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
        wine = load_wine(as_frame=True).frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
        return [(bc, "target", 400), (iris, "target", 100), (wine, "target", 120)]
    diab = load_diabetes(as_frame=True).frame
    cal = fetch_california_housing(as_frame=True).frame.sample(n=1500, random_state=0).reset_index(drop=True)
    return [(diab, "target", 300), (cal, "MedHouseVal", 1000), (diab.iloc[:120], "target", 90)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--out", type=Path, default=HERE.parent / "models")
    args = ap.parse_args()

    for task, name in (("classification", "classifier"), ("regression", "regressor")):
        model = sdm.models.KumoTabular(size=args.size, task=task, device="cpu").eval()
        cases = [capture(model, task, *d) for d in datasets(task)]
        x, y, mask, _ = cases[0]
        print(f"{name}: x {tuple(x.shape)} {x.dtype}, y {tuple(y.shape)} {y.dtype}, mask {tuple(mask.shape)}")

        folder = args.out / f"kumo-tabular-{args.size}-{name}"
        (folder / "onnx").mkdir(parents=True, exist_ok=True)
        path = folder / "onnx" / "model.onnx"
        R = torch.export.Dim("R", min=4, max=65536)
        Rc = torch.export.Dim("Rc", min=2, max=65536)
        C = torch.export.Dim("C", min=1, max=512)
        torch.onnx.export(
            Head(model.models[task]).eval(), (x, y, mask), str(path), dynamo=True, opset_version=20,
            input_names=["x", "y_context", "categorical_mask"], output_names=["logits"],
            dynamic_shapes={"x": {0: R, 1: C}, "y_context": {0: Rc}, "categorical_mask": {0: C}},
            custom_translation_table={torch.ops.aten.remainder.Scalar: rem_scalar},
        )
        # Re-save with the Transformers.js external-data name.
        graph = onnx.load(str(path))
        for stale in (folder / "onnx").glob("model.onnx.data"):
            stale.unlink()
        onnx.save(graph, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                  location="model.onnx_data", size_threshold=1024)

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        worst = 0.0
        for cx, cy, cm, ref in cases:
            (got,) = session.run(None, {"x": cx.numpy(), "y_context": cy.numpy(), "categorical_mask": cm.numpy()})
            diff = float(np.abs(got - ref.float().numpy()).max())
            worst = max(worst, diff)
            print(f"  shape x {tuple(cx.shape)} y {tuple(cy.shape)}: max |diff| {diff:.2e}")
        assert worst <= 1e-3, f"{name} parity failed: {worst}"

        ops = sorted({n.op_type for n in onnx.load(str(path), load_external_data=False).graph.node})
        (folder / "config.json").write_text(json.dumps({
            "model_type": "kumo_tabular",
            "architectures": ["KumoTabular"],
            "kumo_tabular": {
                "source": "nvidia/Kumo-Tabular", "revision": "v1.0.0", "size": args.size, "task": task,
                "num_outputs": 10 if task == "classification" else 999,
                "inputs": "x [R, C] float32 (context rows first), y_context [Rc] "
                          + ("int64 class codes" if task == "classification" else "float32") + ", categorical_mask [C] bool",
            },
            "transformers.js_config": {"use_external_data_format": {"model.onnx": 1}},
        }, indent=2) + "\n")
        print(f"{name}: ok, worst {worst:.2e}; ops: {', '.join(ops)}")


if __name__ == "__main__":
    main()
