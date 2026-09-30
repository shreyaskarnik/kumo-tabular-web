// End-to-end check: fixture CSV -> DuckDB-WASM -> JS preprocessing -> ONNX on
// WebGPU/WASM -> decoded predictions, compared with sdm's (fixtures/*/pred.bin).
import { AutoModel, Tensor, env } from "@huggingface/transformers";
import { ident, loadFile, numericColumn, query, stringColumn } from "./db";
import { classProbabilities, prepare, regressionOutputs, type FeatureColumn } from "./kumo/preprocess";

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = new URL("models/", location.href).href;
const device = (new URLSearchParams(location.search).get("device") ?? "webgpu") as "webgpu" | "wasm";
const log = document.getElementById("log")!;
const out = (s: string) => { log.textContent += s + "\n"; console.log(s); };

(async () => {
  try {
    const results = [];
    for (const name of ["breast-cancer", "iris", "diabetes", "mixed-churn", "mixed-ltv"]) {
      const meta = await (await fetch(`/fixtures/${name}/meta.json`)).json();
      await loadFile("data.csv", new Uint8Array(await (await fetch(`/fixtures/${name}/data.csv`)).arrayBuffer()));
      const feats = Object.keys(meta.stypes).filter((c) => c !== meta.target);
      const cols = feats.map((c) => (meta.stypes[c] === "numerical" ? `CAST(${ident(c)} AS DOUBLE)` : `CAST(${ident(c)} AS VARCHAR)`) + ` AS ${ident(c)}`);
      const t = meta.task === "classification" ? `CAST(${ident(meta.target)} AS VARCHAR)` : `CAST(${ident(meta.target)} AS DOUBLE)`;
      const table = await query(`SELECT ${cols.join(", ")}, ${t} AS __target FROM source`);
      const features: FeatureColumn[] = feats.map((c) =>
        meta.stypes[c] === "numerical"
          ? { name: c, kind: "numerical", values: numericColumn(table, c) }
          : { name: c, kind: "categorical", values: stringColumn(table, c) },
      );
      const target = meta.task === "classification" ? stringColumn(table, "__target") : numericColumn(table, "__target");
      const p = prepare(features, target, meta.task, meta.context);
      const head = meta.task === "classification" ? "classifier" : "regressor";
      const model = await AutoModel.from_pretrained(`kumo-tabular-small-${head}`, { device, dtype: "fp32" } as never);
      const started = performance.now();
      const res = await (model as any)({
        x: new Tensor("float32", p.x, [p.rows, p.columns]),
        y_context: p.yContext instanceof Int32Array ? new Tensor("int32", p.yContext, [p.yContext.length]) : new Tensor("float32", p.yContext, [p.yContext.length]),
        categorical_mask: new Tensor("bool", p.categoricalMask, [p.columns]),
      });
      const ms = performance.now() - started;
      const logits = res.logits.data as Float32Array;
      const q = meta.rows - meta.context;
      const ref = new Float32Array(await (await fetch(`/fixtures/${name}/pred.bin`)).arrayBuffer());
      let worst = 0, flips = 0;
      if (meta.task === "classification") {
        const k = p.classes!.length;
        const probs = classProbabilities(logits, q, k);
        for (let i = 0; i < probs.length; i++) worst = Math.max(worst, Math.abs(probs[i] - ref[i]));
        for (let r = 0; r < q; r++) {
          const am = (a: ArrayLike<number>) => { let b = 0; for (let j = 1; j < k; j++) if (a[r * k + j] > a[r * k + b]) b = j; return b; };
          if (am(probs) !== am(ref)) flips++;
        }
      } else {
        const { point } = regressionOutputs(logits, q, p.target!);
        for (let i = 0; i < q; i++) worst = Math.max(worst, Math.abs(point[i] - ref[i]) / Math.max(1, Math.abs(ref[i])));
      }
      const r = { name, device, task: meta.task, x: [p.rows, p.columns], classesMatch: JSON.stringify(p.classes) === JSON.stringify(meta.classes), maxDiff: worst, flips, ms: Math.round(ms) };
      results.push(r);
      out(JSON.stringify(r));
    }
    out("RESULT " + JSON.stringify(results));
    out("DONE");
  } catch (err) {
    out("ERROR " + ((err as Error)?.stack ?? err));
  }
})();
