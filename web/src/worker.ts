/// <reference lib="webworker" />
// Runs Kumo Tabular (ONNX) off the main thread with Transformers.js.
import { AutoModel, Tensor, env } from "@huggingface/transformers";

export type Head = "classifier" | "regressor";
export type WorkerRequest =
  | { type: "load"; head: Head; modelBase: string; device: "webgpu" | "wasm" }
  | {
      type: "run";
      head: Head;
      x: Float32Array;
      rows: number;
      columns: number;
      yContext: Int32Array | Float32Array;
      categoricalMask: Uint8Array;
    };
export type WorkerResponse =
  | { type: "progress"; head: Head; loaded: number; total: number }
  | { type: "loaded"; head: Head; device: string; ms: number }
  | { type: "result"; head: Head; logits: Float32Array; dims: number[]; ms: number }
  | { type: "error"; message: string };

const post = (message: WorkerResponse, transfer: Transferable[] = []) =>
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(message, transfer);

type Model = (inputs: Record<string, Tensor>) => Promise<{ logits: Tensor }>;
const models: Partial<Record<Head, Promise<{ model: Model; device: string }>>> = {};

async function load(head: Head, modelBase: string, device: "webgpu" | "wasm") {
  const started = performance.now();
  // "hub": the published onnx-community repos; otherwise a URL of local copies.
  const hub = modelBase === "hub";
  if (!hub) {
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = modelBase.endsWith("/") ? modelBase : `${modelBase}/`;
  }
  const files: Record<string, { loaded: number; total: number }> = {};
  const model = (await AutoModel.from_pretrained(hub ? `onnx-community/kumo-tabular-small-${head}-ONNX` : `kumo-tabular-small-${head}`, {
    device,
    dtype: "fp32",
    progress_callback: (info: { status: string; file?: string; loaded?: number; total?: number }) => {
      if (info.status !== "progress" || !info.file) return;
      files[info.file] = { loaded: info.loaded ?? 0, total: info.total ?? 0 };
      const sum = (k: "loaded" | "total") => Object.values(files).reduce((s, f) => s + f[k], 0);
      post({ type: "progress", head, loaded: sum("loaded"), total: sum("total") });
    },
  } as never)) as unknown as Model;
  post({ type: "loaded", head, device, ms: performance.now() - started });
  return { model, device };
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  try {
    if (msg.type === "load") {
      models[msg.head] ??= load(msg.head, msg.modelBase, msg.device);
      await models[msg.head];
      return;
    }
    const entry = models[msg.head];
    if (!entry) throw new Error(`Model ${msg.head} is not loaded.`);
    const { model } = await entry;
    const started = performance.now();
    const out = await model({
      x: new Tensor("float32", msg.x, [msg.rows, msg.columns]),
      y_context:
        msg.yContext instanceof Int32Array
          ? new Tensor("int32", msg.yContext, [msg.yContext.length])
          : new Tensor("float32", msg.yContext, [msg.yContext.length]),
      categorical_mask: new Tensor("bool", msg.categoricalMask, [msg.columns]),
    });
    const logits = Float32Array.from(out.logits.data as Float32Array);
    post({ type: "result", head: msg.head, logits, dims: out.logits.dims, ms: performance.now() - started }, [logits.buffer]);
  } catch (err) {
    post({ type: "error", message: (err as Error)?.message ?? String(err) });
  }
};
