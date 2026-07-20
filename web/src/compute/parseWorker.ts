/**
 * Parse worker (serial, singleton). Runs WASM parse_intern_binary — the
 * expensive pickle decode + frame intern + alloc pairing stage. Emits
 * a compact IR that a layout worker post-processes.
 *
 * Main → Worker: { type: "init", wasmModule }
 * Main → Worker: { type: "parse", rank, file | buffer }
 *                 // File is read inside this worker so large snapshots
 *                 // can report byte progress and avoid main-thread
 *                 // ArrayBuffer ownership churn. display limits are
 *                 // applied later by the layout worker so "show more"
 *                 // avoids reparsing.
 * Worker → Main: { type: "ready" }
 * Worker → Main: { type: "readProgress", rank, loaded, total, done }
 * Worker → Main: { type: "ir", rank, ir }
 * Worker → Main: { type: "error", rank, error }
 */

import { initSync, parse_intern_binary } from "../../../wasm/pkg/memviz_wasm.js";
import { estimateIRBytes, irTransferables } from "./rankIr";

let wasmModule: WebAssembly.Module | null = null;
let ready = false;

function readFileWithProgress(file: File, rank: number): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    const total = file.size || 0;
    reader.onprogress = (ev) => {
      (self as any).postMessage({
        type: "readProgress",
        rank,
        loaded: ev.loaded,
        total: ev.total || total,
        done: false,
      });
    };
    reader.onerror = () => reject(reader.error || new Error("File read failed"));
    reader.onload = () => {
      (self as any).postMessage({
        type: "readProgress",
        rank,
        loaded: total,
        total,
        done: true,
      });
      resolve(reader.result as ArrayBuffer);
    };
    reader.readAsArrayBuffer(file);
  });
}

self.onmessage = async (e: MessageEvent) => {
  const { type } = e.data;

  if (type === "init") {
    try {
      wasmModule = e.data.wasmModule;
      initSync({ module: wasmModule! });
      ready = true;
      (self as any).postMessage({ type: "ready" });
    } catch (err: any) {
      (self as any).postMessage({ type: "error", rank: -1, error: `WASM init: ${err}` });
    }
    return;
  }

  if (type === "parse") {
    const { rank, file } = e.data;
    let { buffer } = e.data;
    try {
      if (!ready || !wasmModule) throw new Error("WASM not initialized");
      if (!buffer && file) buffer = await readFileWithProgress(file, rank);
      if (!buffer) throw new Error("No snapshot buffer");
      const t0 = performance.now();
      const ir = parse_intern_binary(new Uint8Array(buffer), rank);
      const wasmMs = performance.now() - t0;
      (self as any).postMessage(
        { type: "ir", rank, ir, wasmMs, irBytes: estimateIRBytes(ir) },
        irTransferables(ir),
      );
      initSync({ module: wasmModule });
    } catch (err: any) {
      (self as any).postMessage({ type: "error", rank, error: String(err) });
    }
    return;
  }
};
