export interface TopAllocIR {
  idx: number;
  addr: number;
  size: number;
  alloc_us: number;
  free_requested_us: number;
  free_us: number;
  top_frame_idx: number;
  stack_idx: number;
}

export interface BinaryRankIR {
  kind: "memviz-ir-v1";
  metaJson: string;
  allocAddr: Float64Array;
  allocSize: Float64Array;
  allocAllocUs: Float64Array;
  allocFreeRequestedUs: Float64Array;
  allocFreeUs: Float64Array;
  allocTopFrameIdx: Int32Array;
  allocStackIdx: Int32Array;
}

export type RankIR = string | BinaryRankIR;

const BINARY_IR_KIND = "memviz-ir-v1";

export function isBinaryRankIR(ir: unknown): ir is BinaryRankIR {
  return (
    !!ir &&
    typeof ir === "object" &&
    (ir as { kind?: unknown }).kind === BINARY_IR_KIND &&
    typeof (ir as { metaJson?: unknown }).metaJson === "string" &&
    (ir as { allocAddr?: unknown }).allocAddr instanceof Float64Array &&
    (ir as { allocSize?: unknown }).allocSize instanceof Float64Array
  );
}

function allocArrays(ir: BinaryRankIR): (Float64Array | Int32Array)[] {
  return [
    ir.allocAddr,
    ir.allocSize,
    ir.allocAllocUs,
    ir.allocFreeRequestedUs,
    ir.allocFreeUs,
    ir.allocTopFrameIdx,
    ir.allocStackIdx,
  ];
}

export function estimateIRBytes(ir: RankIR): number {
  if (typeof ir === "string") return ir.length;
  return ir.metaJson.length + allocArrays(ir).reduce((sum, arr) => sum + arr.byteLength, 0);
}

export function irTransferables(ir: RankIR): Transferable[] {
  if (!isBinaryRankIR(ir)) return [];
  const out: Transferable[] = [];
  const seen = new Set<ArrayBuffer>();
  for (const arr of allocArrays(ir)) {
    const buffer = arr.buffer as ArrayBuffer;
    if (buffer.byteLength === 0 || seen.has(buffer)) continue;
    seen.add(buffer);
    out.push(buffer);
  }
  return out;
}

export function binaryAllocationCount(ir: BinaryRankIR): number {
  return ir.allocAddr.length;
}

export function binaryTopAllocations(ir: BinaryRankIR, start = 0, end = ir.allocAddr.length): TopAllocIR[] {
  const n = ir.allocAddr.length;
  const lo = Math.max(0, Math.min(n, Math.floor(start)));
  const hi = Math.max(lo, Math.min(n, Math.floor(end)));
  const out: TopAllocIR[] = new Array(hi - lo);
  for (let i = lo; i < hi; i++) {
    out[i - lo] = {
      idx: i,
      addr: ir.allocAddr[i],
      size: ir.allocSize[i],
      alloc_us: ir.allocAllocUs[i],
      free_requested_us: ir.allocFreeRequestedUs[i],
      free_us: ir.allocFreeUs[i],
      top_frame_idx: ir.allocTopFrameIdx[i],
      stack_idx: ir.allocStackIdx[i],
    };
  }
  return out;
}

function normalizeDumpLimit(limit: number | null | undefined, total: number): number {
  if (limit === null) return total;
  return Math.max(0, Math.min(total, Math.floor(limit ?? 1000)));
}

function attachDumpDebug(raw: any, kind: string, total: number, dumped: number, limit: number | null | undefined) {
  raw._debug = {
    ...(raw._debug || {}),
    ir_kind: kind,
    top_allocations_total: total,
    top_allocations_dumped: dumped,
    top_allocations_truncated: dumped < total,
    allocation_limit: limit === undefined ? 1000 : limit,
  };
  return raw;
}

export function rankIRToDebugJson(
  ir: RankIR,
  opts: { allocationLimit?: number | null } = {},
): any {
  if (isBinaryRankIR(ir)) {
    const raw = JSON.parse(ir.metaJson);
    const total = binaryAllocationCount(ir);
    const dumped = normalizeDumpLimit(opts.allocationLimit, total);
    raw.top_allocations = binaryTopAllocations(ir, 0, dumped);
    return attachDumpDebug(raw, ir.kind, total, dumped, opts.allocationLimit);
  }

  const raw = JSON.parse(ir);
  const allAllocs = Array.isArray(raw.top_allocations) ? raw.top_allocations : [];
  const total = allAllocs.length;
  const dumped = normalizeDumpLimit(opts.allocationLimit, total);
  if (dumped < total) raw.top_allocations = allAllocs.slice(0, dumped);
  return attachDumpDebug(raw, "memviz-json-ir", total, Math.min(dumped, total), opts.allocationLimit);
}

export function rankIRToDebugJsonText(
  ir: RankIR,
  opts: { allocationLimit?: number | null } = {},
): string {
  return JSON.stringify(rankIRToDebugJson(ir, opts), null, 2);
}
