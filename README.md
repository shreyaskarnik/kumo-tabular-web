# Kumo Tabular in the browser

Drop a CSV or Parquet file, pick a target column, get zero-shot predictions from
NVIDIA's [Kumo Tabular](https://huggingface.co/nvidia/Kumo-Tabular) (Small, 28M),
with no server. DuckDB-WASM loads and queries the data; the model runs as ONNX
on WebGPU (WASM fallback) through Transformers.js.

## Layout

- `export/`: Python (uv). `export_onnx.py` exports the classifier and regressor
  heads (M1); `fixtures.py` writes golden fixtures from `sdm` (M2).
- `fixtures/`: sdm inputs/outputs per dataset (breast cancer, iris, diabetes,
  two synthetic mixed tables with categoricals, NaNs and a constant column).
- `web/`: Vite + TypeScript app. `src/kumo/preprocess.ts` ports sdm's default
  recipe (estimator 0, without the random augmentations); `src/db.ts` is the
  DuckDB layer; `src/worker.ts` runs the model.
- `models/` (not committed): the exported ONNX repos.

## Checks

- Export: ONNX Runtime vs PyTorch, max |diff| 1.25e-4 (classifier) and
  4.4e-5 (regressor) at three shapes each.
- Preprocessing: `npm test` compares processed `x`, masks, targets, class order
  and decoded predictions with the fixtures (x within 1e-4).
- End to end (`web/e2e.html`): CSV -> DuckDB -> JS preprocessing -> WebGPU ->
  predictions match sdm within 6e-6, no top class changed.

## Develop

```bash
cd export && uv venv .venv && uv pip install -e ./sdm-src onnx onnxruntime onnxscript scikit-learn pandas
.venv/bin/python export_onnx.py && .venv/bin/python fixtures.py
cd ../web && npm install && npm test && npm run dev
```

## Limits (v1)

One estimator (sdm's default), up to 10 classes, context sampled to 10,000
rows, queries capped at 5,000. Weights are OpenMDW-1.1; code is Apache-2.0.
