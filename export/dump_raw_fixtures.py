"""Save the inner-model tensors captured from sdm as binary fixtures for the browser check."""
import json
from pathlib import Path
import numpy as np
import sdm
from export_onnx import capture, datasets

out = Path(__file__).parent.parent / "fixtures" / "raw"
out.mkdir(parents=True, exist_ok=True)
index = []
for task, name in (("classification", "classifier"), ("regression", "regressor")):
    model = sdm.models.KumoTabular(size="small", task=task, device="cpu").eval()
    for i, d in enumerate(datasets(task)):
        x, y, mask, ref = capture(model, task, *d)
        stem = f"{name}-{i}"
        x.numpy().astype(np.float32).tofile(out / f"{stem}.x.bin")
        (y.numpy().astype(np.int32) if task == "classification" else y.numpy().astype(np.float32)).tofile(out / f"{stem}.y.bin")
        mask.numpy().astype(np.uint8).tofile(out / f"{stem}.mask.bin")
        ref.float().numpy().astype(np.float32).tofile(out / f"{stem}.ref.bin")
        index.append({"stem": stem, "head": name, "x": list(x.shape), "y": list(y.shape), "ydtype": "int32" if task == "classification" else "float32", "out": list(ref.shape)})
(out / "index.json").write_text(json.dumps(index, indent=1))
print(json.dumps(index))
