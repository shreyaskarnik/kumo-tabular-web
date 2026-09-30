// Probe: run the exported heads on the raw sdm fixtures in the browser and
// compare with PyTorch's outputs. ?device=webgpu|wasm
import { AutoModel, Tensor, env } from "@huggingface/transformers";

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "/models/";

const log = document.getElementById("log")!;
const out = (s: string) => { log.textContent += s + "\n"; console.log(s); };
const device = (new URLSearchParams(location.search).get("device") ?? "webgpu") as "webgpu" | "wasm";

type Entry = { stem: string; head: string; x: number[]; y: number[]; ydtype: string; out: number[] };
const bin = async (path: string) => new Uint8Array(await (await fetch(path)).arrayBuffer());

(async () => {
  try {
    const index: Entry[] = await (await fetch("/fixtures/raw/index.json")).json();
    const results: unknown[] = [];
    const models: Record<string, any> = {};
    for (const e of index) {
      const t0 = performance.now();
      models[e.head] ??= await AutoModel.from_pretrained(`kumo-tabular-small-${e.head}`, { device, dtype: "fp32" } as never);
      const loadMs = performance.now() - t0;
      const x = new Float32Array((await bin(`/fixtures/raw/${e.stem}.x.bin`)).buffer);
      const yRaw = (await bin(`/fixtures/raw/${e.stem}.y.bin`)).buffer;
      const y = e.ydtype === "int32" ? new Int32Array(yRaw) : new Float32Array(yRaw);
      const mask = await bin(`/fixtures/raw/${e.stem}.mask.bin`);
      const ref = new Float32Array((await bin(`/fixtures/raw/${e.stem}.ref.bin`)).buffer);
      const inputs = () => ({
        x: new Tensor("float32", x, e.x),
        y_context: new Tensor(e.ydtype as "int32", y, e.y),
        categorical_mask: new Tensor("bool", mask, [e.x[1]]),
      });
      const times: number[] = [];
      let got: Float32Array | null = null;
      for (let i = 0; i < 3; i++) {
        const t = performance.now();
        const res = await models[e.head](inputs());
        got = res.logits.data as Float32Array;
        times.push(performance.now() - t);
      }
      let diff = 0;
      for (let i = 0; i < ref.length; i++) diff = Math.max(diff, Math.abs(got![i] - ref[i]));
      const r = { stem: e.stem, device, x: e.x, loadMs: Math.round(loadMs), firstMs: Math.round(times[0]), warmMs: Math.round(Math.min(times[1], times[2])), maxDiff: diff, finite: got!.every(Number.isFinite) };
      results.push(r);
      out(JSON.stringify(r));
    }
    out("RESULT " + JSON.stringify(results));
    out("DONE");
  } catch (err) {
    out("ERROR " + ((err as Error)?.stack ?? err));
  }
})();
