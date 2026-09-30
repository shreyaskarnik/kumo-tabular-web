/**
 * Kumo Tabular's default preprocessing recipe (sdm/models/kumo/tabular/recipe.py),
 * estimator 0, without the random augmentations (FlipSign, ShuffleColumns,
 * ShuffleCategories). Everything is fitted on the context rows and applied to
 * all rows. Checked value by value against sdm in tests/preprocess.test.ts.
 */

export type Task = "classification" | "regression";

export type FeatureColumn =
  | { name: string; kind: "numerical"; values: ArrayLike<number> } // NaN = missing
  | { name: string; kind: "categorical"; values: ReadonlyArray<string | null> };

export type Prepared = {
  /** [rows, columns] float32, context rows first. */
  x: Float32Array;
  rows: number;
  columns: number;
  columnNames: string[];
  /** One per column: whether it came from a categorical feature. */
  categoricalMask: Uint8Array;
  /** Class codes (int32) or standardized targets (float32) of the context rows. */
  yContext: Int32Array | Float32Array;
  /** Classification: class labels in code order. */
  classes: string[] | null;
  /** Regression: how to undo the target standardization. */
  target: { mean: number; scale: number } | null;
};

const MAX_COLUMNS = 500;
const COUNT_MIN_CARDINALITY = 50;

/** A column of float64 values, all rows; statistics use the first `n` (context). */
type Col = { name: string; values: Float64Array; categorical: boolean };

const isFiniteNumber = (v: number) => Number.isFinite(v);

/** DropConstantColumns(threshold=1): keep if any context value differs from the first. */
function keepColumn(values: Float64Array, n: number): boolean {
  if (n <= 1) return true;
  const first = values[0];
  let different = false;
  let allNaN = true;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (!(v === first)) different = true; // NaN never equals anything
    if (!Number.isNaN(v)) allNaN = false;
  }
  return different && !allNaN;
}

/** Standardize(eps): population statistics over finite context values. */
function standardize(values: Float64Array, n: number, eps: number): { mean: number; scale: number } {
  let count = 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (isFiniteNumber(v)) {
      count++;
      sum += v;
    }
  }
  const mean = count ? sum / count : 0;
  let sq = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (isFiniteNumber(v)) sq += (v - mean) * (v - mean);
  }
  const variance = count ? sq / count : 0;
  let scale = Math.sqrt(variance);
  if (eps === 0) {
    // Constant columns keep unit scale (sdm's _constant_feature_mask).
    if (scale === 0 || !Number.isFinite(scale)) scale = 1;
  } else {
    scale += eps;
  }
  return { mean, scale };
}

/** ClipSigma(threshold): two-pass z-score bounds, then logarithmic soft clipping. */
function clipSigma(values: Float64Array, n: number, threshold: number): void {
  let count = 0;
  let sum = 0;
  for (let i = 0; i < n; i++) if (isFiniteNumber(values[i])) (count++, (sum += values[i]));
  let mean = count ? sum / count : 0;
  let sq = 0;
  for (let i = 0; i < n; i++) if (isFiniteNumber(values[i])) sq += (values[i] - mean) ** 2;
  let std = Math.max(Math.sqrt(sq / Math.max(count - 1, 1)), 1e-6);
  const lower0 = mean - threshold * std;
  const upper0 = mean + threshold * std;

  let kept = 0;
  let keptSum = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (isFiniteNumber(v) && v >= lower0 && v <= upper0) (kept++, (keptSum += v));
  }
  if (kept > 0) {
    const keptMean = keptSum / kept;
    let keptSq = 0;
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (isFiniteNumber(v) && v >= lower0 && v <= upper0) keptSq += (v - keptMean) ** 2;
    }
    mean = keptMean;
    std = Math.max(Math.sqrt(keptSq / Math.max(kept - 1, 1)), 1e-6);
  }
  const lower = mean - threshold * std;
  const upper = mean + threshold * std;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isNaN(v)) continue;
    const logAbs = Math.log1p(Math.abs(v));
    values[i] = Math.min(logAbs + upper, Math.max(lower - logAbs, v));
  }
}

/** The numerical chain: DropConstantColumns, Standardize(1e-6), Clip(±100), Identity, ClipSigma(4). */
function numericalChain(columns: Col[], n: number): Col[] {
  const kept = columns.filter((c) => keepColumn(c.values, n));
  for (const c of kept) {
    const { mean, scale } = standardize(c.values, n, 1e-6);
    const v = c.values;
    for (let i = 0; i < v.length; i++) {
      if (Number.isNaN(v[i])) continue;
      v[i] = Math.min(100, Math.max(-100, (v[i] - mean) / scale));
    }
    clipSigma(v, n, 4);
  }
  return kept;
}

const compareValues = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

/**
 * AlignCategories(sort_by="value"), AddCategoryCounts(50), ToNumerical: count
 * columns first, then the code columns (missing or unseen = -1).
 */
function categoricalColumns(
  features: Extract<FeatureColumn, { kind: "categorical" }>[],
  rows: number,
  n: number,
): Col[] {
  const counts: Col[] = [];
  const codes: Col[] = [];
  for (const f of features) {
    const seen = new Set<string>();
    for (let i = 0; i < n; i++) {
      const v = f.values[i];
      if (v !== null) seen.add(v);
    }
    const categories = [...seen].sort(compareValues);
    const index = new Map(categories.map((c, i) => [c, i]));
    const code = new Float64Array(rows);
    for (let i = 0; i < rows; i++) {
      const v = f.values[i];
      code[i] = v === null ? -1 : (index.get(v) ?? -1);
    }
    codes.push({ name: f.name, values: code, categorical: true });
    if (categories.length > COUNT_MIN_CARDINALITY) {
      // log1p of the context count per code; negative codes share one slot.
      const tally = new Float64Array(categories.length + 1);
      for (let i = 0; i < n; i++) tally[code[i] >= 0 ? code[i] : categories.length]++;
      const count = new Float64Array(rows);
      for (let i = 0; i < rows; i++) count[i] = Math.log1p(tally[code[i] >= 0 ? code[i] : categories.length]);
      counts.push({ name: `${f.name}__count`, values: count, categorical: false });
    }
  }
  return [...counts, ...codes];
}

/**
 * Build the model inputs. Rows `0..contextRows-1` are context (their target
 * must be present); the rest are queries.
 */
export function prepare(
  features: FeatureColumn[],
  target: ReadonlyArray<string | null> | ArrayLike<number>,
  task: Task,
  contextRows: number,
): Prepared {
  const rows = features.length ? features[0].values.length : (target as ArrayLike<unknown>).length;
  const n = contextRows;

  const numerical: Col[] = [];
  const categorical: Extract<FeatureColumn, { kind: "categorical" }>[] = [];
  for (const f of features) {
    if (f.kind === "numerical") {
      numerical.push({ name: f.name, values: Float64Array.from(f.values), categorical: false });
    } else {
      categorical.push(f);
    }
  }
  const columns = [
    ...numericalChain(numerical, n),
    ...numericalChain(categoricalColumns(categorical, rows, n), n),
  ].slice(0, MAX_COLUMNS);

  const x = new Float32Array(rows * columns.length);
  columns.forEach((c, j) => {
    for (let i = 0; i < rows; i++) x[i * columns.length + j] = c.values[i];
  });

  let yContext: Int32Array | Float32Array;
  let classes: string[] | null = null;
  let targetScale: Prepared["target"] = null;
  if (task === "classification") {
    // AlignCategories(sort_by="code"): classes in order of first appearance.
    const labels = target as ReadonlyArray<string | null>;
    classes = [];
    const index = new Map<string, number>();
    yContext = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const label = labels[i];
      if (label === null) throw new Error(`Context row ${i} has no target.`);
      if (!index.has(label)) (index.set(label, classes.length), classes.push(label));
      yContext[i] = index.get(label)!;
    }
  } else {
    const values = Float64Array.from(target as ArrayLike<number>).subarray(0, n);
    const { mean, scale } = standardize(values, n, 0);
    targetScale = { mean, scale };
    yContext = Float32Array.from(values, (v) => (v - mean) / scale);
  }

  return {
    x,
    rows,
    columns: columns.length,
    columnNames: columns.map((c) => c.name),
    categoricalMask: Uint8Array.from(columns, (c) => (c.categorical ? 1 : 0)),
    yContext,
    classes,
    target: targetScale,
  };
}

/** Classification: softmax over the first K of the 10 logits, per query row. */
export function classProbabilities(logits: Float32Array, queryRows: number, numClasses: number): Float32Array {
  const width = logits.length / queryRows;
  const out = new Float32Array(queryRows * numClasses);
  for (let r = 0; r < queryRows; r++) {
    let max = -Infinity;
    for (let k = 0; k < numClasses; k++) max = Math.max(max, logits[r * width + k]);
    let sum = 0;
    for (let k = 0; k < numClasses; k++) sum += out[r * numClasses + k] = Math.exp(logits[r * width + k] - max);
    for (let k = 0; k < numClasses; k++) out[r * numClasses + k] /= sum;
  }
  return out;
}

/**
 * Regression: sort the 999 quantiles per row and undo the target
 * standardization. Returns the quantiles and the point estimate (their mean).
 */
export function regressionOutputs(
  quantiles: Float32Array,
  queryRows: number,
  target: { mean: number; scale: number },
): { point: Float64Array; quantiles: Float64Array } {
  const width = quantiles.length / queryRows;
  const point = new Float64Array(queryRows);
  const sorted = new Float64Array(quantiles.length);
  for (let r = 0; r < queryRows; r++) {
    const row = Float64Array.from(quantiles.subarray(r * width, (r + 1) * width)).sort();
    let sum = 0;
    for (let q = 0; q < width; q++) {
      const v = row[q] * target.scale + target.mean;
      sorted[r * width + q] = v;
      sum += v;
    }
    point[r] = sum / width;
  }
  return { point, quantiles: sorted };
}
