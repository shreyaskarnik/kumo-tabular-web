// M3: the JS preprocessing against sdm's golden fixtures (export/fixtures.py).
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { classProbabilities, prepare, regressionOutputs, type FeatureColumn } from "../src/kumo/preprocess";

const ROOT = join(__dirname, "..", "..", "fixtures");
const names = readdirSync(ROOT).filter((n) => n !== "raw");

/** Exact bytes of a file (a Node Buffer may be a view into a larger pool). */
const bytes = (path: string) => new Uint8Array(readFileSync(path)).buffer;
const f32 = (path: string) => new Float32Array(bytes(path));

/** pandas' to_csv output: no quoting in these fixtures; empty = missing. */
function readCsv(path: string): { header: string[]; rows: string[][] } {
  const [head, ...lines] = readFileSync(path, "utf8").trimEnd().split("\n");
  return { header: head.split(","), rows: lines.map((l) => l.split(",")) };
}

describe.each(names)("%s", (name) => {
  const dir = join(ROOT, name);
  const meta = JSON.parse(readFileSync(join(dir, "meta.json"), "utf8"));
  const { header, rows } = readCsv(join(dir, "data.csv"));
  const column = (col: string) => rows.map((r) => r[header.indexOf(col)]);

  const features: FeatureColumn[] = header
    .filter((h) => h !== meta.target)
    .map((h) =>
      meta.stypes[h] === "numerical"
        ? { name: h, kind: "numerical", values: Float64Array.from(column(h), (v) => (v === "" ? NaN : Number(v))) }
        : { name: h, kind: "categorical", values: column(h).map((v) => (v === "" ? null : v)) },
    );
  const target =
    meta.task === "classification"
      ? column(meta.target).map((v) => (v === "" ? null : v))
      : Float64Array.from(column(meta.target), Number);
  const prepared = prepare(features, target, meta.task, meta.context);

  it("builds x like sdm", () => {
    const x = f32(join(dir, "x.bin"));
    expect([prepared.rows, prepared.columns]).toEqual(meta.x);
    let worst = 0;
    for (let i = 0; i < x.length; i++) {
      const a = x[i], b = prepared.x[i];
      if (Number.isNaN(a) || Number.isNaN(b)) expect(Number.isNaN(a) && Number.isNaN(b)).toBe(true);
      else worst = Math.max(worst, Math.abs(a - b));
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it("builds the categorical mask and y like sdm", () => {
    expect(Array.from(prepared.categoricalMask)).toEqual(Array.from(new Uint8Array(readFileSync(join(dir, "mask.bin")))));
    const raw = bytes(join(dir, "y.bin"));
    if (meta.task === "classification") {
      expect(Array.from(prepared.yContext)).toEqual(Array.from(new Int32Array(raw)));
      expect(prepared.classes).toEqual(meta.classes);
    } else {
      const y = new Float32Array(raw);
      prepared.yContext.forEach((v, i) => expect(Math.abs(v - y[i])).toBeLessThan(1e-5));
    }
  });

  it("decodes sdm's logits to sdm's predictions", () => {
    const logits = f32(join(dir, "logits.bin"));
    const pred = f32(join(dir, "pred.bin"));
    const queryRows = meta.rows - meta.context;
    if (meta.task === "classification") {
      const probs = classProbabilities(logits, queryRows, meta.classes.length);
      probs.forEach((p, i) => expect(Math.abs(p - pred[i])).toBeLessThan(1e-5));
    } else {
      const { point } = regressionOutputs(logits, queryRows, prepared.target!);
      point.forEach((p, i) => expect(Math.abs(p - pred[i]) / Math.max(1, Math.abs(pred[i]))).toBeLessThan(1e-4));
    }
  });
});
