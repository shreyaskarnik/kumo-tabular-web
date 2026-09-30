import { describe, exportCsv, ident, loadFile, numericColumn, query, stringColumn, tableFromRows, type ColumnInfo } from "./db";
import { classProbabilities, prepare, regressionOutputs, type FeatureColumn, type Task } from "./kumo/preprocess";
import type { Head, WorkerRequest, WorkerResponse } from "./worker";

const MAX_CONTEXT = 10_000;
const MAX_QUERY = 5_000;
const MAX_CLASSES = 10;
const SEED = 42;
// onnx-community/kumo-tabular-small-{classifier,regressor}-ONNX; `?models=local` serves ./models instead.
const MODEL_BASE: string =
  new URLSearchParams(location.search).get("models") === "local" ? new URL("models/", location.href).href : (import.meta.env.VITE_MODEL_BASE ?? "hub");

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const esc = (s: unknown) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? Number(v.toPrecision(digits + 1)).toLocaleString() : "—");

// ---------------------------------------------------------------- state
let columns: ColumnInfo[] = [];
let rowCount = 0;
let sampleTarget: string | null = null;
const excluded = new Set<string>();
let device: "webgpu" | "wasm" = "wasm";

// ---------------------------------------------------------------- worker
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
let pending: { resolve: (m: WorkerResponse) => void; reject: (e: Error) => void } | null = null;
let onProgress: ((loaded: number, total: number) => void) | null = null;
worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
  const msg = event.data;
  if (msg.type === "progress") return onProgress?.(msg.loaded, msg.total);
  if (msg.type === "error") pending?.reject(new Error(msg.message));
  else pending?.resolve(msg);
  pending = null;
};
function call(request: WorkerRequest, transfer: Transferable[] = []): Promise<WorkerResponse> {
  return new Promise((resolve, reject) => {
    pending = { resolve, reject };
    worker.postMessage(request, transfer);
  });
}
const loadedHeads = new Set<Head>();

// ---------------------------------------------------------------- data
async function openData(name: string, data: File | Uint8Array, target: string | null) {
  status(`Reading ${name}…`);
  sampleTarget = target;
  await loadFile(name, data);
  ($("sql") as HTMLTextAreaElement).value = "SELECT * FROM source";
  await applySql(name);
}

async function applySql(label?: string) {
  const sql = ($("sql") as HTMLTextAreaElement).value.trim().replace(/;+\s*$/, "");
  try {
    await query(`CREATE OR REPLACE TABLE data AS SELECT row_number() OVER () AS __row, * FROM (${sql})`);
    const info = await describe("data");
    rowCount = info.rows;
    columns = info.columns.filter((c) => c.name !== "__row");
    excluded.clear();
    $("data-summary").textContent = `${label ? `${label} · ` : ""}${rowCount.toLocaleString()} rows · ${columns.length} columns`;
    renderSetup();
    $("setup-card").hidden = false;
    $("results-card").hidden = true;
    status("");
  } catch (err) {
    status((err as Error).message, true);
  }
}

// ---------------------------------------------------------------- setup
const targetSelect = () => $("target") as HTMLSelectElement;
const taskSelect = () => $("task") as HTMLSelectElement;
const modeSelect = () => $("mode") as HTMLSelectElement;
const targetInfo = () => columns.find((c) => c.name === targetSelect().value)!;

function defaultTask(c: ColumnInfo): Task {
  if (c.kind === "categorical") return "classification";
  return c.integral && c.distinct <= MAX_CLASSES ? "classification" : "regression";
}

function renderSetup() {
  const usable = columns.filter((c) => c.kind !== "unsupported");
  const select = targetSelect();
  select.innerHTML = usable.map((c) => `<option>${esc(c.name)}</option>`).join("");
  const preferred = usable.find((c) => c.name === sampleTarget) ?? usable[usable.length - 1];
  if (preferred) select.value = preferred.name;
  onTargetChange();
}

function onTargetChange() {
  const t = targetInfo();
  if (!t) return;
  taskSelect().value = defaultTask(t);
  modeSelect().value = t.nulls > 0 ? "missing" : "holdout";
  renderColumns();
  renderSplitNote();
}

function renderColumns() {
  const target = targetSelect().value;
  $("columns").innerHTML = columns
    .map((c) => {
      const isTarget = c.name === target;
      const off = c.kind === "unsupported" || excluded.has(c.name);
      const kind = c.kind === "unsupported" ? `${c.type.toLowerCase()} · skipped` : c.kind;
      return `<label class="chip${isTarget ? " target" : off ? " off" : ""}">
        <input type="checkbox" data-col="${esc(c.name)}" ${isTarget || off ? "" : "checked"} ${isTarget || c.kind === "unsupported" ? "disabled" : ""}/>
        ${esc(c.name)} <span class="kind">${isTarget ? "target" : kind}</span></label>`;
    })
    .join("");
}

function features(): ColumnInfo[] {
  const target = targetSelect().value;
  return columns.filter((c) => c.name !== target && c.kind !== "unsupported" && !excluded.has(c.name));
}

function renderSplitNote() {
  const t = targetInfo();
  const labeled = rowCount - t.nulls;
  let text: string;
  if (modeSelect().value === "missing") {
    text = t.nulls
      ? `${labeled.toLocaleString()} labeled rows teach the model; it predicts the ${t.nulls.toLocaleString()} rows where ${t.name} is empty.`
      : `Every row has a ${t.name}, so there is nothing to predict here. Choose "Hold out 20%" to score the model.`;
  } else {
    const held = Math.round(labeled * 0.2);
    text = `About ${(labeled - held).toLocaleString()} labeled rows teach the model; about ${held.toLocaleString()} held-out rows are predicted and scored against their real ${t.name}.`;
  }
  if (labeled > MAX_CONTEXT) text += ` Context is sampled down to ${MAX_CONTEXT.toLocaleString()} rows.`;
  if (taskSelect().value === "classification" && t.distinct > MAX_CLASSES)
    text += ` ${t.name} has ${t.distinct} values; this demo handles up to ${MAX_CLASSES} classes.`;
  $("split-note").textContent = text;
}

// ---------------------------------------------------------------- run
function status(text: string, error = false) {
  const el = $("status");
  el.textContent = text;
  el.className = error ? "error" : "muted";
}

function setProgress(fraction: number | null) {
  const bar = $("progress");
  bar.hidden = fraction === null;
  (bar.firstElementChild as HTMLElement).style.width = `${Math.round((fraction ?? 0) * 100)}%`;
}

type Run = {
  task: Task;
  target: string;
  classes: string[] | null;
  rows: number[];
  actual: (string | number | null)[];
  prediction: (string | number)[];
  confidence: Float64Array;
  probs: Float32Array | null;
  interval: [Float64Array, Float64Array] | null;
  contextRows: number;
  baseline: number;
  timings: { load: number | null; prep: number; infer: number };
};

async function run() {
  const button = $("run") as HTMLButtonElement;
  button.disabled = true;
  try {
    const task = taskSelect().value as Task;
    const t = targetInfo();
    const feats = features();
    if (!feats.length) throw new Error("Pick at least one feature column.");
    const target = ident(t.name);
    const hash = `hash(__row + ${SEED}) % 1000`;
    const labeled = `${target} IS NOT NULL`;
    const [contextWhere, queryWhere] =
      modeSelect().value === "missing"
        ? [labeled, `${target} IS NULL`]
        : [`${labeled} AND ${hash} >= 200`, `${labeled} AND ${hash} < 200`];

    status("Selecting rows…");
    await query(`CREATE OR REPLACE TABLE run_rows AS
      (SELECT __row, 0 AS role FROM data WHERE ${contextWhere} ORDER BY hash(__row * 7919 + ${SEED}) LIMIT ${MAX_CONTEXT})
      UNION ALL
      (SELECT __row, 1 AS role FROM data WHERE ${queryWhere} ORDER BY __row LIMIT ${MAX_QUERY})`);
    const select = feats
      .map((c) => (c.kind === "numerical" ? `CAST(${ident(c.name)} AS DOUBLE)` : `CAST(${ident(c.name)} AS VARCHAR)`) + ` AS ${ident(c.name)}`)
      .join(", ");
    const targetExpr = task === "classification" ? `CAST(${target} AS VARCHAR)` : `CAST(${target} AS DOUBLE)`;
    const table = await query(`SELECT d.__row, r.role, ${select}, ${targetExpr} AS __target
      FROM data d JOIN run_rows r USING (__row) ORDER BY r.role, d.__row`);
    const roles = Array.from({ length: table.numRows }, (_, i) => Number(table.getChild("role")!.get(i)));
    const contextRows = roles.filter((r) => r === 0).length;
    const queryRows = table.numRows - contextRows;
    if (contextRows < 2) throw new Error("Need at least two labeled rows.");
    if (!queryRows) throw new Error("No rows to predict. Try holding out 20%.");

    const started = performance.now();
    const featureColumns: FeatureColumn[] = feats.map((c) =>
      c.kind === "numerical"
        ? { name: c.name, kind: "numerical", values: numericColumn(table, c.name) }
        : { name: c.name, kind: "categorical", values: stringColumn(table, c.name) },
    );
    const targetValues = task === "classification" ? stringColumn(table, "__target") : numericColumn(table, "__target");
    const prepared = prepare(featureColumns, targetValues, task, contextRows);
    if (prepared.classes && prepared.classes.length > MAX_CLASSES)
      throw new Error(`${t.name} has ${prepared.classes.length} classes in the labeled rows; up to ${MAX_CLASSES} are supported.`);
    if (prepared.classes && prepared.classes.length < 2) throw new Error("The labeled rows have only one class.");
    if (!prepared.columns) throw new Error("Every feature is constant in the labeled rows.");
    const prep = performance.now() - started;

    const head: Head = task === "classification" ? "classifier" : "regressor";
    let load: number | null = null;
    if (!loadedHeads.has(head)) {
      status("Downloading the model (about 110 MB, cached after the first time)…");
      setProgress(0);
      onProgress = (loaded, total) => total && setProgress(loaded / total);
      const loadedMsg = await call({ type: "load", head, modelBase: MODEL_BASE, device });
      onProgress = null;
      if (loadedMsg.type === "loaded") load = loadedMsg.ms;
      loadedHeads.add(head);
    }
    setProgress(null);
    status(`Predicting ${queryRows.toLocaleString()} rows from ${contextRows.toLocaleString()} examples on ${device === "webgpu" ? "WebGPU" : "WASM"}…`);
    const result = await call(
      { type: "run", head, x: prepared.x, rows: prepared.rows, columns: prepared.columns, yContext: prepared.yContext, categoricalMask: prepared.categoricalMask },
      [prepared.x.buffer],
    );
    if (result.type !== "result") throw new Error("Unexpected worker reply.");

    const rows = Array.from({ length: queryRows }, (_, i) => Number(table.getChild("__row")!.get(contextRows + i)));
    const actual = Array.from({ length: queryRows }, (_, i) => {
      const v = targetValues[contextRows + i];
      return typeof v === "number" ? (Number.isNaN(v) ? null : v) : v;
    });
    let out: Run;
    if (task === "classification") {
      const classes = prepared.classes!;
      const probs = classProbabilities(result.logits, queryRows, classes.length);
      const prediction: string[] = [];
      const confidence = new Float64Array(queryRows);
      for (let r = 0; r < queryRows; r++) {
        let best = 0;
        for (let k = 1; k < classes.length; k++) if (probs[r * classes.length + k] > probs[r * classes.length + best]) best = k;
        prediction.push(classes[best]);
        confidence[r] = probs[r * classes.length + best];
      }
      const counts = new Map<string, number>();
      for (let i = 0; i < contextRows; i++) counts.set(String(targetValues[i]), (counts.get(String(targetValues[i])) ?? 0) + 1);
      const majority = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
      const known = actual.filter((a) => a !== null);
      const baseline = known.length ? known.filter((a) => a === majority).length / known.length : NaN;
      out = { task, target: t.name, classes, rows, actual, prediction, confidence, probs, interval: null, contextRows, baseline, timings: { load, prep, infer: result.ms } };
    } else {
      const { point, quantiles } = regressionOutputs(result.logits, queryRows, prepared.target!);
      const width = quantiles.length / queryRows;
      const lo = new Float64Array(queryRows);
      const hi = new Float64Array(queryRows);
      for (let r = 0; r < queryRows; r++) {
        lo[r] = quantiles[r * width + 49]; // q050
        hi[r] = quantiles[r * width + 949]; // q950
      }
      const confidence = Float64Array.from(point, (_, r) => hi[r] - lo[r]);
      let mean = 0;
      for (let i = 0; i < contextRows; i++) mean += (targetValues[i] as number) / contextRows;
      const known = actual.map((a, i) => [a, i] as const).filter(([a]) => a !== null);
      const baseline = known.length ? Math.sqrt(known.reduce((s, [a]) => s + ((a as number) - mean) ** 2, 0) / known.length) : NaN;
      out = { task, target: t.name, classes: null, rows, actual, prediction: Array.from(point), confidence, probs: null, interval: [lo, hi], contextRows, baseline, timings: { load, prep, infer: result.ms } };
    }
    await storePredictions(out);
    renderResults(out);
    status("");
  } catch (err) {
    setProgress(null);
    status((err as Error).message, true);
  } finally {
    button.disabled = false;
  }
}

async function storePredictions(r: Run) {
  const rows = r.rows.map((row, i) => {
    const rec: Record<string, unknown> = { __row: row, prediction: r.prediction[i] };
    if (r.task === "classification") {
      rec.confidence = Number(r.confidence[i].toFixed(4));
      r.classes!.forEach((c, k) => (rec[`p_${c}`] = Number(r.probs![i * r.classes!.length + k].toFixed(4))));
    } else {
      rec.p05 = r.interval![0][i];
      rec.p95 = r.interval![1][i];
    }
    return rec;
  });
  await tableFromRows("predictions", rows);
  await query(`CREATE OR REPLACE VIEW results AS SELECT d.*, p.* EXCLUDE (__row) FROM data d JOIN predictions p USING (__row) ORDER BY d.__row`);
}

// ---------------------------------------------------------------- results
function metric(value: string, label: string) {
  return `<div class="metric"><b>${value}</b><span>${label}</span></div>`;
}

function renderResults(r: Run) {
  $("results-card").hidden = false;
  const known = r.actual.map((a, i) => [a, i] as const).filter(([a]) => a !== null);
  const tiles: string[] = [];
  if (r.task === "classification") {
    if (known.length) {
      const acc = known.filter(([a, i]) => a === r.prediction[i]).length / known.length;
      let logLoss = 0;
      for (const [a, i] of known) {
        const k = r.classes!.indexOf(String(a));
        logLoss -= Math.log(Math.max(k >= 0 ? r.probs![i * r.classes!.length + k] : 0, 1e-12)) / known.length;
      }
      tiles.push(metric(`${(acc * 100).toFixed(1)}%`, `accuracy on ${known.length} held-out rows`));
      tiles.push(metric(`${(r.baseline * 100).toFixed(1)}%`, "always guessing the most common class"));
      tiles.push(metric(fmt(logLoss), "log loss"));
    } else {
      tiles.push(metric(r.rows.length.toLocaleString(), "rows predicted"));
      const mean = r.confidence.reduce((s, v) => s + v, 0) / r.confidence.length;
      tiles.push(metric(`${(mean * 100).toFixed(1)}%`, "average confidence"));
    }
  } else if (known.length) {
    let se = 0, ae = 0, mean = 0, inside = 0;
    for (const [a] of known) mean += (a as number) / known.length;
    let ss = 0;
    for (const [a, i] of known) {
      const e = (r.prediction[i] as number) - (a as number);
      se += e * e;
      ae += Math.abs(e);
      ss += ((a as number) - mean) ** 2;
      if ((a as number) >= r.interval![0][i] && (a as number) <= r.interval![1][i]) inside++;
    }
    tiles.push(metric(fmt(Math.sqrt(se / known.length)), `RMSE on ${known.length} held-out rows`));
    tiles.push(metric(fmt(r.baseline), "RMSE of predicting the mean"));
    tiles.push(metric(fmt(1 - se / ss), "R²"));
    tiles.push(metric(`${((inside / known.length) * 100).toFixed(0)}%`, "inside the 90% interval"));
  } else {
    tiles.push(metric(r.rows.length.toLocaleString(), "rows predicted"));
  }
  tiles.push(metric(`${fmt(r.timings.infer, 2)} ms`, `model, ${device === "webgpu" ? "WebGPU" : "WASM"}, ${r.contextRows.toLocaleString()} examples`));
  $("metrics").innerHTML = tiles.join("");
  $("timing").textContent =
    (r.timings.load !== null ? `model load ${fmt(r.timings.load / 1000, 2)} s · ` : "") + `preprocessing ${fmt(r.timings.prep, 2)} ms`;

  $("chart").innerHTML = r.task === "classification" ? classChart(r, known.length > 0) : regressionChart(r, known.length > 0);
  renderPreview(r);
  const q = $("query") as HTMLTextAreaElement;
  q.value =
    r.task === "classification"
      ? `SELECT prediction, count(*) AS rows, round(avg(confidence), 3) AS avg_confidence\nFROM results GROUP BY 1 ORDER BY rows DESC`
      : `-- the rows the model is least sure about\nSELECT * EXCLUDE (__row) FROM results ORDER BY p95 - p05 DESC LIMIT 20`;
  runQuery();
  $("results-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

function classChart(r: Run, scored: boolean): string {
  const classes = r.classes!;
  if (scored) {
    const m = classes.map(() => classes.map(() => 0));
    r.actual.forEach((a, i) => {
      const ai = classes.indexOf(String(a));
      if (ai >= 0) m[ai][classes.indexOf(String(r.prediction[i]))]++;
    });
    const max = Math.max(1, ...m.flat());
    return `<h3>Confusion matrix <span class="muted">(rows: actual, columns: predicted)</span></h3>
      <div class="table-wrap"><table class="cm"><tr><th></th>${classes.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>
      ${m.map((row, i) => `<tr><th>${esc(classes[i])}</th>${row.map((v, j) => `<td style="background:color-mix(in srgb, ${i === j ? "var(--accent)" : "var(--bad)"} ${Math.round((v / max) * 70)}%, transparent)">${v}</td>`).join("")}</tr>`).join("")}
      </table></div>`;
  }
  const counts = classes.map((c) => r.prediction.filter((p) => p === c).length);
  const max = Math.max(1, ...counts);
  const bar = 26;
  return `<h3>Predicted classes</h3><svg viewBox="0 0 360 ${classes.length * bar + 8}" role="img" aria-label="Predicted class counts">
    ${classes.map((c, i) => `<text x="0" y="${i * bar + 17}" fill="currentColor" font-size="12">${esc(c).slice(0, 14)}</text>
      <rect x="110" y="${i * bar + 5}" width="${(counts[i] / max) * 200}" height="16" rx="4" fill="var(--accent)"/>
      <text x="${116 + (counts[i] / max) * 200}" y="${i * bar + 17}" fill="currentColor" font-size="12">${counts[i]}</text>`).join("")}
  </svg>`;
}

function regressionChart(r: Run, scored: boolean): string {
  const W = 360, H = 300, P = 36;
  const pred = r.prediction as number[];
  if (!scored) {
    const lo = Math.min(...pred), hi = Math.max(...pred), bins = 24;
    const counts = new Array(bins).fill(0);
    pred.forEach((v) => counts[Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo || 1)) * bins))]++);
    const max = Math.max(...counts);
    return `<h3>Predicted ${esc(r.target)}</h3><svg viewBox="0 0 ${W} ${H}">${counts.map((c, i) => `<rect x="${P + (i * (W - P)) / bins}" y="${H - P - (c / max) * (H - 2 * P)}" width="${(W - P) / bins - 2}" height="${(c / max) * (H - 2 * P)}" fill="var(--accent)" rx="2"/>`).join("")}
      <text x="${P}" y="${H - 12}" font-size="11" fill="currentColor">${fmt(lo)}</text><text x="${W - 4}" y="${H - 12}" font-size="11" fill="currentColor" text-anchor="end">${fmt(hi)}</text></svg>`;
  }
  const pts = r.actual.map((a, i) => [a as number, pred[i], r.interval![0][i], r.interval![1][i]] as const).filter(([a]) => a !== null);
  const all = pts.flatMap(([a, p]) => [a, p]);
  const lo = Math.min(...all), hi = Math.max(...all);
  const sx = (v: number) => P + ((v - lo) / (hi - lo || 1)) * (W - P - 8);
  const sy = (v: number) => H - P - ((v - lo) / (hi - lo || 1)) * (H - P - 8);
  return `<h3>Predicted vs actual <span class="muted">(bars: 90% interval)</span></h3>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted versus actual">
    <line x1="${sx(lo)}" y1="${sy(lo)}" x2="${sx(hi)}" y2="${sy(hi)}" stroke="var(--muted)" stroke-dasharray="4 4"/>
    ${pts.map(([a, p, l, u]) => `<line x1="${sx(a)}" x2="${sx(a)}" y1="${sy(l)}" y2="${sy(u)}" stroke="var(--accent)" stroke-opacity="0.25"/><circle cx="${sx(a)}" cy="${sy(p)}" r="2.6" fill="var(--accent)"/>`).join("")}
    <text x="${W / 2}" y="${H - 6}" font-size="11" fill="currentColor" text-anchor="middle">actual</text>
    <text x="10" y="${H / 2}" font-size="11" fill="currentColor" transform="rotate(-90 10 ${H / 2})" text-anchor="middle">predicted</text>
    <text x="${P}" y="${H - 20}" font-size="10" fill="var(--muted)">${fmt(lo)}</text><text x="${W - 8}" y="${H - 20}" font-size="10" fill="var(--muted)" text-anchor="end">${fmt(hi)}</text>
    </svg>`;
}

function renderPreview(r: Run) {
  const scored = r.actual.some((a) => a !== null);
  const head =
    r.task === "classification"
      ? `<tr><th>row</th>${scored ? `<th>actual</th>` : ""}<th>prediction</th><th>confidence</th></tr>`
      : `<tr><th>row</th>${scored ? `<th>actual</th>` : ""}<th>prediction</th><th>90% interval</th></tr>`;
  const body = r.rows
    .slice(0, 300)
    .map((row, i) => {
      const actual = scored ? `<td class="${r.task === "regression" ? "num" : ""}">${esc(r.task === "regression" ? fmt(r.actual[i] as number) : r.actual[i])}</td>` : "";
      if (r.task === "classification") {
        const wrong = scored && r.actual[i] !== r.prediction[i];
        const c = r.confidence[i];
        return `<tr><td>${row}</td>${actual}<td class="${wrong ? "wrong" : ""}">${esc(r.prediction[i])}</td><td><span class="conf"><i style="width:${Math.round(c * 100)}%"></i></span>${(c * 100).toFixed(0)}%</td></tr>`;
      }
      return `<tr><td>${row}</td>${actual}<td class="num">${fmt(r.prediction[i] as number)}</td><td class="num">${fmt(r.interval![0][i])} – ${fmt(r.interval![1][i])}</td></tr>`;
    })
    .join("");
  ($("preview") as HTMLTableElement).innerHTML = head + body;
}

async function runQuery() {
  const out = $("query-out") as HTMLTableElement;
  try {
    const table = await query(($("query") as HTMLTextAreaElement).value);
    const names = table.schema.fields.map((f) => f.name);
    const rows = table.toArray().slice(0, 500).map((r) => r.toJSON());
    out.innerHTML =
      `<tr>${names.map((n) => `<th>${esc(n)}</th>`).join("")}</tr>` +
      rows
        .map((r) => `<tr>${names.map((n) => {
          const v = r[n];
          return typeof v === "number" ? `<td class="num">${fmt(v)}</td>` : `<td>${esc(typeof v === "bigint" ? v.toString() : v)}</td>`;
        }).join("")}</tr>`)
        .join("");
  } catch (err) {
    out.innerHTML = `<tr><td class="error">${esc((err as Error).message)}</td></tr>`;
  }
}

// ---------------------------------------------------------------- wiring
async function init() {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
    if (gpu && (await gpu.requestAdapter())) device = "webgpu";
  } catch {
    device = "wasm";
  }
  const badge = $("device-badge");
  badge.textContent = device === "webgpu" ? "WebGPU" : "WASM (no WebGPU: slower)";
  badge.classList.toggle("ok", device === "webgpu");

  const file = $("file") as HTMLInputElement;
  file.addEventListener("change", () => file.files?.[0] && openData(file.files[0].name, file.files[0], null));
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => (e.preventDefault(), drop.classList.add("over")));
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    const f = e.dataTransfer?.files?.[0];
    if (f) openData(f.name, f, null);
  });
  document.querySelectorAll<HTMLButtonElement>("[data-sample]").forEach((b) =>
    b.addEventListener("click", async () => {
      const name = b.dataset.sample!;
      const bytes = new Uint8Array(await (await fetch(new URL(`samples/${name}`, location.href))).arrayBuffer());
      await openData(name, bytes, b.dataset.target ?? null);
    }),
  );
  $("apply-sql").addEventListener("click", () => applySql());
  targetSelect().addEventListener("change", onTargetChange);
  taskSelect().addEventListener("change", renderSplitNote);
  modeSelect().addEventListener("change", renderSplitNote);
  $("columns").addEventListener("change", (e) => {
    const input = e.target as HTMLInputElement;
    const name = input.dataset.col!;
    if (input.checked) excluded.delete(name);
    else excluded.add(name);
    input.closest(".chip")!.classList.toggle("off", !input.checked);
  });
  $("run").addEventListener("click", run);
  $("run-query").addEventListener("click", runQuery);
  $("download").addEventListener("click", async () => {
    const bytes = await exportCsv("SELECT * EXCLUDE (__row) FROM results");
    const url = URL.createObjectURL(new Blob([bytes.slice()], { type: "text/csv" }));
    Object.assign(document.createElement("a"), { href: url, download: "results.csv" }).click();
    URL.revokeObjectURL(url);
  });
  const sample = new URLSearchParams(location.search).get("sample");
  const button = sample && document.querySelector<HTMLButtonElement>(`[data-sample^="${CSS.escape(sample)}"]`);
  if (button) button.click();
}

init();
