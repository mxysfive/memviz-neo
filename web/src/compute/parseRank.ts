// Layout-worker entry: takes the Rust-emitted IR JSON (frames/stack
// pools, segments, allocation records) and produces the final RankData
// the main thread renders. Polygon layout runs here in pure JS so the
// N layout workers never touch WASM — their JS heap is GC'd, unlike
// WASM linear memory which is grow-only.
//
// parseRank is intentionally a thin pipeline: each stage below is a
// named helper that consumes the previous stage's output. Read from
// top to bottom.

import type { RankSummary, SegmentInfo, TopAllocation, FrameRecord } from "../types/snapshot";
import type { TimelineAlloc, TimelineTimeAxis } from "../types/timeline";
import { STRIP_FLOATS } from "../types/timeline";
import type { Anomaly } from "./anomalies";
import { detectAnomalies } from "./anomalies";
import type { Allocation, RankData, SegmentRow, SegmentAlloc, TraceEvent, FlameData, FlameNode } from "./index";
import { blockColor } from "./palette";
import { eventIdxAt } from "./eventTimes";
import { isInternalFrame } from "../utils";
import {
  binaryAllocationCount,
  binaryTopAllocations,
  isBinaryRankIR,
  type BinaryRankIR,
  type RankIR,
  type TopAllocIR,
} from "./rankIr";

export interface ParseResult {
  data: RankData;
}

/** Output of `packStrips` — downstream stages (segment rows, RankData
 *  assembly) consume these aggregates together so we pass one struct. */
interface PackedTimeline {
  stripBuffer: Float32Array;        // GPU buffer, source X axis since t_min
  stripBufferEvent: Float32Array;   // GPU buffer, event-ordinal axis
  eventTimes: Float64Array;         // sorted unique event times (relative to t_min)
  timelineAllocs: TimelineAlloc[];  // one per alloc; stripOffset/Count point into stripBuffer
  /** [r, g, b] per alloc, flat. Segment rows reuse this so colors match. */
  allocColors: Float32Array;
  maxBytesFull: number;             // largest y + size seen across all strips
  stripCount: number;
}

interface ParseRankOptions {
  layoutLimit?: number;
}

interface HiddenStripIR {
  tStart: number;
  tEnd: number;
  yOffset: number;
  size: number;
  count: number;
}

// Port of the Rust build_layout (removed from WASM). O(N²) over the
// currently-rendered allocation subset.
// Output: flat array of [li, t_start, t_end, y_offset] quadruples.
function buildLayout(allocs: TopAllocIR[], tMax: number): Float64Array {
  const n = allocs.length;
  if (n === 0) return new Float64Array(0);

  // Event list: (time, et, li, size). et=1 alloc, et=0 free.
  const evCap = n * 2;
  const evTime = new Float64Array(evCap);
  const evEt = new Uint8Array(evCap);
  const evLi = new Int32Array(evCap);
  const evSz = new Float64Array(evCap);
  let evN = 0;
  for (let li = 0; li < n; li++) {
    const a = allocs[li];
    evTime[evN] = a.alloc_us; evEt[evN] = 1; evLi[evN] = li; evSz[evN] = a.size; evN++;
    if (a.free_us !== -1) {
      evTime[evN] = a.free_us; evEt[evN] = 0; evLi[evN] = li; evSz[evN] = a.size; evN++;
    }
  }
  // Sort by time asc, then et asc so frees at equal time come first
  // (matches Rust's tuple-sort of (time, et, ...) where free=0, alloc=1).
  const orderArr: number[] = new Array(evN);
  for (let i = 0; i < evN; i++) orderArr[i] = i;
  orderArr.sort((a, b) => {
    const d = evTime[a] - evTime[b];
    if (d !== 0) return d;
    return evEt[a] - evEt[b];
  });

  // Stack of live ids + sizes, parallel arrays. pos[li] = index into stack or -1.
  const skId = new Int32Array(n);
  const skSz = new Float64Array(n);
  let skLen = 0;
  const pos = new Int32Array(n).fill(-1);
  const tSt = new Float64Array(n);
  const y = new Float64Array(n);
  const act = new Uint8Array(n);
  let stot = 0;

  // Output grows; size unknown upfront. Keep it as one flat typed buffer
  // to avoid four push-heavy JS arrays in show-all layouts.
  let outCap = Math.max(16, n * 2);
  let outN = 0;
  let out = new Float64Array(outCap * 4);
  const pushStrip = (liOut: number, ts: number, te: number, yo: number) => {
    if (outN >= outCap) {
      outCap *= 2;
      const next = new Float64Array(outCap * 4);
      next.set(out);
      out = next;
    }
    const off = outN * 4;
    out[off] = liOut;
    out[off + 1] = ts;
    out[off + 2] = te;
    out[off + 3] = yo;
    outN++;
  };

  for (let k = 0; k < evN; k++) {
    const i = orderArr[k];
    const time = evTime[i];
    const et = evEt[i];
    const li = evLi[i];
    const sz = evSz[i];
    if (et === 1) {
      y[li] = stot;
      tSt[li] = time;
      pos[li] = skLen;
      skId[skLen] = li;
      skSz[skLen] = sz;
      skLen++;
      act[li] = 1;
      stot += sz;
    } else {
      const p = pos[li];
      if (p === -1) continue;
      if (tSt[li] < time) {
        pushStrip(li, tSt[li], time, y[li]);
      }
      act[li] = 0;
      pos[li] = -1;
      const freed = skSz[p];
      for (let j = p; j < skLen - 1; j++) {
        skId[j] = skId[j + 1];
        skSz[j] = skSz[j + 1];
      }
      skLen--;
      stot -= freed;
      // Update positions + emit closing strips + shift y down by freed.
      for (let j = p; j < skLen; j++) {
        const ai = skId[j];
        pos[ai] = j;
        const oy = y[ai];
        if (tSt[ai] < time) {
          pushStrip(ai, tSt[ai], time, oy);
        }
        tSt[ai] = time;
        y[ai] = oy - freed;
      }
    }
  }
  // Close any still-live strips to t_max.
  for (let li = 0; li < n; li++) {
    if (act[li] && tSt[li] < tMax) {
      pushStrip(li, tSt[li], tMax, y[li]);
    }
  }
  return out.slice(0, outN * 4);
}

function buildHiddenStrips(
  detailAllocs: TopAllocIR[],
  hiddenAllocs: TopAllocIR[],
  timeMin: number,
  timeMax: number,
): HiddenStripIR[] {
  if (hiddenAllocs.length === 0) return [];

  type Ev = {
    time: number;
    detailDelta: number;
    hiddenDelta: number;
    hiddenCountDelta: number;
  };
  const events: Ev[] = [];
  const addAllocEvents = (a: TopAllocIR, hidden: boolean) => {
    const endRaw = a.free_us === -1 ? timeMax : a.free_us;
    const start = Math.max(timeMin, Math.min(timeMax, a.alloc_us));
    const end = Math.max(timeMin, Math.min(timeMax, endRaw));
    if (start >= end) return;
    if (hidden) {
      events.push({ time: start, detailDelta: 0, hiddenDelta: a.size, hiddenCountDelta: 1 });
      events.push({ time: end, detailDelta: 0, hiddenDelta: -a.size, hiddenCountDelta: -1 });
    } else {
      events.push({ time: start, detailDelta: a.size, hiddenDelta: 0, hiddenCountDelta: 0 });
      events.push({ time: end, detailDelta: -a.size, hiddenDelta: 0, hiddenCountDelta: 0 });
    }
  };

  for (const a of detailAllocs) addAllocEvents(a, false);
  for (const a of hiddenAllocs) addAllocEvents(a, true);
  if (events.length === 0) return [];

  events.sort((a, b) => a.time - b.time);
  const out: HiddenStripIR[] = [];
  let detailLive = 0;
  let hiddenLive = 0;
  let hiddenCount = 0;
  let i = 0;
  while (i < events.length) {
    const t = events[i].time;
    while (i < events.length && events[i].time === t) {
      detailLive += events[i].detailDelta;
      hiddenLive += events[i].hiddenDelta;
      hiddenCount += events[i].hiddenCountDelta;
      i++;
    }
    const next = i < events.length ? events[i].time : timeMax;
    if (next <= t || hiddenLive <= 0 || hiddenCount <= 0) continue;
    const yOffset = Math.max(0, detailLive);
    const size = Math.max(0, hiddenLive);
    const count = Math.max(1, hiddenCount);
    const prev = out[out.length - 1];
    if (
      prev &&
      prev.tEnd === t &&
      prev.yOffset === yOffset &&
      prev.size === size &&
      prev.count === count
    ) {
      prev.tEnd = next;
    } else {
      out.push({ tStart: t, tEnd: next, yOffset, size, count });
    }
  }
  return out;
}

function buildHiddenStripsFromBinary(
  detailAllocs: TopAllocIR[],
  ir: BinaryRankIR,
  hiddenStart: number,
  timeMin: number,
  timeMax: number,
): HiddenStripIR[] {
  const total = binaryAllocationCount(ir);
  if (hiddenStart >= total) return [];

  type Ev = {
    time: number;
    detailDelta: number;
    hiddenDelta: number;
    hiddenCountDelta: number;
  };
  const events: Ev[] = [];
  const add = (allocUs: number, freeUs: number, size: number, hidden: boolean) => {
    const endRaw = freeUs === -1 ? timeMax : freeUs;
    const start = Math.max(timeMin, Math.min(timeMax, allocUs));
    const end = Math.max(timeMin, Math.min(timeMax, endRaw));
    if (start >= end) return;
    if (hidden) {
      events.push({ time: start, detailDelta: 0, hiddenDelta: size, hiddenCountDelta: 1 });
      events.push({ time: end, detailDelta: 0, hiddenDelta: -size, hiddenCountDelta: -1 });
    } else {
      events.push({ time: start, detailDelta: size, hiddenDelta: 0, hiddenCountDelta: 0 });
      events.push({ time: end, detailDelta: -size, hiddenDelta: 0, hiddenCountDelta: 0 });
    }
  };

  for (const a of detailAllocs) add(a.alloc_us, a.free_us, a.size, false);
  for (let i = hiddenStart; i < total; i++) {
    add(ir.allocAllocUs[i], ir.allocFreeUs[i], ir.allocSize[i], true);
  }
  if (events.length === 0) return [];

  events.sort((a, b) => a.time - b.time);
  const out: HiddenStripIR[] = [];
  let detailLive = 0;
  let hiddenLive = 0;
  let hiddenCount = 0;
  let i = 0;
  while (i < events.length) {
    const t = events[i].time;
    while (i < events.length && events[i].time === t) {
      detailLive += events[i].detailDelta;
      hiddenLive += events[i].hiddenDelta;
      hiddenCount += events[i].hiddenCountDelta;
      i++;
    }
    const next = i < events.length ? events[i].time : timeMax;
    if (next <= t || hiddenLive <= 0 || hiddenCount <= 0) continue;
    const yOffset = Math.max(0, detailLive);
    const size = Math.max(0, hiddenLive);
    const count = Math.max(1, hiddenCount);
    const prev = out[out.length - 1];
    if (
      prev &&
      prev.tEnd === t &&
      prev.yOffset === yOffset &&
      prev.size === size &&
      prev.count === count
    ) {
      prev.tEnd = next;
    } else {
      out.push({ tStart: t, tEnd: next, yOffset, size, count });
    }
  }
  return out;
}

/**
 * Turn the layout output into GPU-ready strip buffers + per-alloc
 * TimelineAllocs. Colors are assigned here (stable per-call-site hue +
 * per-instance lightness shift) so segment rows can re-use them for
 * cross-view color consistency.
 *
 * Also synthesises the "event ordinal" buffer for the alternate X-axis
 * mode: same geometry, t columns replaced by event indices.
 */
function packStrips(
  topAllocsIR: TopAllocIR[],
  stripsFlat: Float64Array,
  hiddenStrips: HiddenStripIR[],
  baseline: number,
  timeMin: number,
  timeMax: number,
  eventTimes: Float64Array,
): PackedTimeline {
  const n = topAllocsIR.length;
  const detailStripCount = stripsFlat.length / 4;
  const totalStrips = detailStripCount + hiddenStrips.length;

  // Bucket strips by owning alloc with typed arrays. The previous
  // number[][] form created one JS array per allocation and pushed every
  // strip through it; show-all snapshots spend most of that time in
  // object allocation / GC rather than numeric work.
  const stripCounts = new Uint32Array(n);
  for (let s = 0; s < detailStripCount; s++) {
    stripCounts[stripsFlat[s * 4] as number]++;
  }
  const stripOffsetsByAlloc = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) {
    stripOffsetsByAlloc[i + 1] = stripOffsetsByAlloc[i] + stripCounts[i];
  }
  const stripCursor = new Uint32Array(stripOffsetsByAlloc);
  const stripOrder = new Uint32Array(detailStripCount);
  for (let s = 0; s < detailStripCount; s++) {
    const li = stripsFlat[s * 4] as number;
    stripOrder[stripCursor[li]++] = s;
  }

  const stripBuffer = new Float32Array(totalStrips * STRIP_FLOATS);
  const timelineAllocs: TimelineAlloc[] = new Array(n + hiddenStrips.length);
  const allocColors = new Float32Array(n * 3);
  // maxBytes starts at baseline since in-window strips get shifted up
  // by that much (the Y axis reflects absolute GPU bytes).
  let maxBytesFull = baseline;
  // Per-hueKey instance counter so repeated allocs from the same call
  // site get lightness-shifted shades inside the same color family.
  const instanceCount = new Map<number, number>();
  let writeIdx = 0;

  for (let i = 0; i < n; i++) {
    const a = topAllocsIR[i];
    // Prefer top_frame_idx for hue (user-code line); fall back to
    // stack_idx when the allocator stack has no .py frame so
    // un-attributed allocs still fan out rather than all turning gray.
    const hueKey = a.top_frame_idx >= 0 ? a.top_frame_idx : 0x100000 + a.stack_idx;
    const inst = instanceCount.get(hueKey) || 0;
    instanceCount.set(hueKey, inst + 1);
    const [r, g, bl] = blockColor(hueKey, inst);
    allocColors[i * 3] = r;
    allocColors[i * 3 + 1] = g;
    allocColors[i * 3 + 2] = bl;
    const sz = a.size;
    const startStripIdx = writeIdx;
    const pEnd = stripOffsetsByAlloc[i + 1];
    for (let p = stripOffsetsByAlloc[i]; p < pEnd; p++) {
      const s = stripOrder[p];
      const tStart = stripsFlat[s * 4 + 1];
      const tEnd = stripsFlat[s * 4 + 2];
      const yOff = stripsFlat[s * 4 + 3];
      const off = writeIdx * STRIP_FLOATS;
      stripBuffer[off] = tStart - timeMin;
      stripBuffer[off + 1] = tEnd - timeMin;
      stripBuffer[off + 2] = yOff + baseline;
      stripBuffer[off + 3] = sz;
      stripBuffer[off + 4] = r;
      stripBuffer[off + 5] = g;
      stripBuffer[off + 6] = bl;
      const top = yOff + baseline + sz;
      if (top > maxBytesFull) maxBytesFull = top;
      writeIdx++;
    }
    const alive = a.free_us === -1;
    const freeUs = alive ? timeMax : a.free_us;
    timelineAllocs[i] = {
      addr: a.addr,
      size: a.size,
      alloc_us: a.alloc_us,
      free_requested_us: a.free_requested_us,
      free_us: freeUs,
      alive,
      top_frame_idx: a.top_frame_idx,
      stack_idx: a.stack_idx,
      idx: i,
      stripOffset: startStripIdx,
      stripCount: stripCounts[i],
    };
  }

  const hiddenColor: [number, number, number] = [0.42, 0.44, 0.47];
  for (let i = 0; i < hiddenStrips.length; i++) {
    const h = hiddenStrips[i];
    const allocIdx = n + i;
    const off = writeIdx * STRIP_FLOATS;
    stripBuffer[off] = h.tStart - timeMin;
    stripBuffer[off + 1] = h.tEnd - timeMin;
    stripBuffer[off + 2] = h.yOffset + baseline;
    stripBuffer[off + 3] = h.size;
    stripBuffer[off + 4] = hiddenColor[0];
    stripBuffer[off + 5] = hiddenColor[1];
    stripBuffer[off + 6] = hiddenColor[2];
    const top = h.yOffset + baseline + h.size;
    if (top > maxBytesFull) maxBytesFull = top;
    timelineAllocs[allocIdx] = {
      addr: -(i + 1),
      size: h.size,
      alloc_us: h.tStart,
      free_requested_us: -1,
      free_us: h.tEnd,
      alive: false,
      top_frame_idx: -1,
      stack_idx: -1,
      idx: allocIdx,
      stripOffset: writeIdx,
      stripCount: 1,
      isHidden: true,
      hiddenCount: h.count,
      hiddenReason: "More allocation details exist than the current display budget can expand.",
    };
    writeIdx++;
  }

  // Event-axis buffer: same geometry as stripBuffer but with t_start /
  // t_end remapped to eventTimes indices. One pass, O(strips × log(events)).
  const stripBufferEvent = new Float32Array(stripBuffer.length);
  for (let s = 0; s < totalStrips; s++) {
    const off = s * STRIP_FLOATS;
    stripBufferEvent[off]     = eventIdxAt(eventTimes, stripBuffer[off]);
    stripBufferEvent[off + 1] = eventIdxAt(eventTimes, stripBuffer[off + 1]);
    stripBufferEvent[off + 2] = stripBuffer[off + 2];
    stripBufferEvent[off + 3] = stripBuffer[off + 3];
    stripBufferEvent[off + 4] = stripBuffer[off + 4];
    stripBufferEvent[off + 5] = stripBuffer[off + 5];
    stripBufferEvent[off + 6] = stripBuffer[off + 6];
  }

  return {
    stripBuffer,
    stripBufferEvent,
    eventTimes,
    timelineAllocs,
    allocColors,
    maxBytesFull,
    stripCount: totalStrips,
  };
}

/**
 * Bucket every rendered alloc into the segment whose [address, address+size)
 * range contains it. Empty segments still appear as rows so the
 * allocator layout stays visible even before allocs arrive.
 */
function buildSegmentRows(
  segments: SegmentInfo[],
  topAllocsIR: TopAllocIR[],
  timelineAllocs: TimelineAlloc[],
  allocColors: Float32Array,
): SegmentRow[] {
  const segsByAddr = segments.slice().sort((a, b) => a.address - b.address);
  const rows = new Map<number, SegmentRow>();
  for (const seg of segsByAddr) {
    rows.set(seg.address, {
      segmentAddr: seg.address,
      segmentType: seg.segment_type,
      totalSize: seg.total_size,
      allocs: [],
    });
  }
  function findSegmentIdx(addr: number): number {
    let lo = 0, hi = segsByAddr.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const seg = segsByAddr[mid];
      if (addr < seg.address) hi = mid - 1;
      else if (addr >= seg.address + seg.total_size) lo = mid + 1;
      else return mid;
    }
    return -1;
  }

  for (let i = 0; i < topAllocsIR.length; i++) {
    const a = topAllocsIR[i];
    const segIdx = findSegmentIdx(a.addr);
    if (segIdx < 0) continue;
    const seg = segsByAddr[segIdx];
    const sa: SegmentAlloc = {
      addr: a.addr,
      offsetInSeg: a.addr - seg.address,
      size: a.size,
      alloc_us: a.alloc_us,
      free_us: timelineAllocs[i].free_us,
      top_frame_idx: a.top_frame_idx,
      color: [allocColors[i * 3], allocColors[i * 3 + 1], allocColors[i * 3 + 2]],
    };
    rows.get(seg.address)!.allocs.push(sa);
  }

  // Biggest cached segments first — matches the Memory Timeline's
  // convention of putting the largest, most-significant data at the top.
  return [...rows.values()].sort((a, b) => b.totalSize - a.totalSize);
}

/**
 * Build a prefix trie of call stacks weighted by size × lifetime
 * (bytes·time-unit of memory pressure), then DFS-flatten into a flat node list
 * the flame-graph view can draw directly. Allocator-internal + C++
 * frames get filtered out so the tree shows human-readable PyTorch
 * code paths.
 */
function buildFlameGraph(
  topAllocsIR: TopAllocIR[],
  stackPool: Uint32Array[],
  framePool: FrameRecord[],
  timeMax: number,
): FlameData {
  interface Trie { frameIdx: number; weight: number; kids: Map<number, Trie>; }
  const root: Trie = { frameIdx: -1, weight: 0, kids: new Map() };

  for (const a of topAllocsIR) {
    const lifetime = a.free_us === -1 ? (timeMax - a.alloc_us) : (a.free_us - a.alloc_us);
    if (lifetime <= 0) continue;
    const weight = a.size * lifetime;
    const stack = stackPool[a.stack_idx];
    if (!stack) continue;
    // PyTorch records frames leaf-first; flip to root-first so the
    // flame graph reads naturally (top-level caller at the bottom).
    root.weight += weight;
    let cursor = root;
    for (let k = stack.length - 1; k >= 0; k--) {
      const fi = stack[k];
      const f = framePool[fi];
      if (!f || isInternalFrame(f)) continue;
      let child = cursor.kids.get(fi);
      if (!child) {
        child = { frameIdx: fi, weight: 0, kids: new Map() };
        cursor.kids.set(fi, child);
      }
      child.weight += weight;
      cursor = child;
    }
  }

  const nodes: FlameNode[] = [];
  let maxDepth = 0;
  (function dfs(node: Trie, depth: number, xStart: number) {
    nodes.push({ frameIdx: node.frameIdx, depth, weight: node.weight, xStart });
    if (depth > maxDepth) maxDepth = depth;
    const kids = [...node.kids.values()].sort((a, b) => b.weight - a.weight);
    let cursor = xStart;
    for (const kid of kids) {
      dfs(kid, depth + 1, cursor);
      cursor += kid.weight;
    }
  })(root, 0, 0);

  return { nodes, totalWeight: root.weight, maxDepth };
}

/**
 * Build the Top Allocs list from the same time-series IR the Flamegraph
 * aggregates over. Rows carry stack_idx so the UI can filter by
 * "stack contains frame X" when drilled into a flame node.
 * Sorted by memory pressure (size × lifetime) desc.
 */
function extractTopAllocations(
  topAllocsIR: TopAllocIR[],
  timeMax: number,
): TopAllocation[] {
  const out: TopAllocation[] = topAllocsIR.map((a) => {
    const lifetime = a.free_us === -1 ? timeMax - a.alloc_us : a.free_us - a.alloc_us;
    return {
      address: a.addr,
      size: a.size,
      alloc_us: a.alloc_us,
      free_us: a.free_us,
      lifetime_us: lifetime,
      top_frame_idx: a.top_frame_idx,
      stack_idx: a.stack_idx,
    };
  });
  out.sort((a, b) => b.size * b.lifetime_us - a.size * a.lifetime_us);
  return out;
}

export function parseRank(ir: RankIR, _rank: number, opts: ParseRankOptions = {}): ParseResult {
  const binaryIR = isBinaryRankIR(ir) ? ir : null;
  const raw = JSON.parse(binaryIR ? binaryIR.metaJson : (ir as string));
  const summary: RankSummary = raw.summary;

  // ---- Pools + raw inputs ----
  const rawFramePool: [string, string, number][] = raw.frame_pool || [];
  const framePool: FrameRecord[] = rawFramePool.map(([name, filename, line]) => ({ name, filename, line }));
  const rawStackPool: number[][] = raw.stack_pool || [];
  const stackPool: Uint32Array[] = rawStackPool.map((arr) => Uint32Array.from(arr));

  const traceEvents: TraceEvent[] = (raw.trace_events || []).map((e: any) => ({
    action: String(e.action || ""),
    addr: Number(e.addr || 0),
    size: Number(e.size || 0),
    stream: Number(e.stream || 0),
    time_us: Number(e.time_us || 0),
    top_frame_idx: Number(e.top_frame_idx ?? -1),
    stack_idx: Number(e.stack_idx ?? -1),
  }));
  const timeMin: number = raw.timeline.time_min;
  const timeMax: number = raw.timeline.time_max;
  const timeAxis: TimelineTimeAxis = raw.timeline.time_axis || "time_us";
  // Pre-window allocations not freed in window — aggregated, not
  // per-alloc. Drawn as an opaque band at the bottom so the y axis
  // reflects real memory usage instead of "delta from window start".
  const baseline: number = raw.timeline.baseline || 0;
  const eventSet = new Set<number>();
  eventSet.add(0);
  eventSet.add(timeMax - timeMin);
  for (const e of traceEvents) eventSet.add(e.time_us - timeMin);
  const eventTimes = Float64Array.from([...eventSet].sort((a, b) => a - b));

  const layoutLimit = Math.max(0, Math.floor(opts.layoutLimit ?? 20000));
  const jsonTopAllocations: TopAllocIR[] = raw.top_allocations || [];
  const totalAllocations = binaryIR
    ? binaryAllocationCount(binaryIR)
    : jsonTopAllocations.length;
  const renderedAllocationCount = layoutLimit > 0
    ? Math.min(layoutLimit, totalAllocations)
    : totalAllocations;
  const topAllocsIR = binaryIR
    ? binaryTopAllocations(binaryIR, 0, renderedAllocationCount)
    : (layoutLimit > 0
        ? jsonTopAllocations.slice(0, layoutLimit)
        : jsonTopAllocations);
  const hiddenAllocsIR = !binaryIR && layoutLimit > 0
    ? jsonTopAllocations.slice(layoutLimit)
    : [];
  const hiddenAllocationCount = Math.max(0, totalAllocations - topAllocsIR.length);

  // ---- Segments ----
  const segments: SegmentInfo[] = (raw.segments || []).map((s: any) => ({
    address: s.address,
    total_size: s.total_size,
    allocated_size: s.allocated_size,
    segment_type: s.segment_type,
    stream: Number(s.stream || 0),
    blocks: (s.blocks || []).map((b: any) => ({
      address: b.address,
      size: b.size,
      state: b.state,
      offset_in_segment: b.offset_in_segment,
      top_frame_idx: b.top_frame_idx,
    })),
  }));
  segments.sort((a, b) => b.total_size - a.total_size);

  // ---- Anomaly detection over the currently-rendered cohort ----
  const allocations: Allocation[] = topAllocsIR.map((a) => ({
    addr: a.addr,
    size: a.size,
    alloc_us: a.alloc_us,
    free_requested_us: a.free_requested_us,
    free_us: a.free_us,
    top_frame_idx: a.top_frame_idx,
    stack_idx: a.stack_idx,
  }));
  const anomalies: Anomaly[] = detectAnomalies(allocations, timeMax, timeAxis);

  // Detail lookup: keyed by "addr-alloc_us" so address reuse (PyTorch
  // re-allocates freed GPU memory) doesn't collide.
  const stackByIdentity = new Map<
    string,
    { stack_idx: number; size: number; alloc_us: number; free_us: number; top_frame_idx: number }
  >();
  for (const a of allocations) {
    stackByIdentity.set(`${a.addr}-${a.alloc_us}`, {
      stack_idx: a.stack_idx,
      size: a.size,
      alloc_us: a.alloc_us,
      free_us: a.free_us,
      top_frame_idx: a.top_frame_idx,
    });
  }

  // ---- Layout → strip packing → segment rows → flame graph ----
  const stripsFlat = buildLayout(topAllocsIR, timeMax);
  const hiddenStrips = binaryIR && layoutLimit > 0
    ? buildHiddenStripsFromBinary(topAllocsIR, binaryIR, renderedAllocationCount, timeMin, timeMax)
    : buildHiddenStrips(topAllocsIR, hiddenAllocsIR, timeMin, timeMax);
  const packed = packStrips(topAllocsIR, stripsFlat, hiddenStrips, baseline, timeMin, timeMax, eventTimes);
  const segmentRows = buildSegmentRows(segments, topAllocsIR, packed.timelineAllocs, packed.allocColors);
  const flame = buildFlameGraph(topAllocsIR, stackPool, framePool, timeMax);
  const topAllocations = extractTopAllocations(topAllocsIR, timeMax);

  const data: RankData = {
    summary,
    segments,
    topAllocations: topAllocations.slice(0, 100),
    timeline: {
      usage_series: [],
      annotations: [],
      time_min: timeMin,
      time_max: timeMax,
      time_axis: timeAxis,
      peak_bytes: raw.timeline.peak_bytes,
      allocation_count: raw.timeline.allocation_count,
      rendered_allocation_count: topAllocsIR.length,
      hidden_allocation_count: hiddenAllocationCount,
      hidden_strip_count: hiddenStrips.length,
      baseline,
    },
    timelineAllocs: packed.timelineAllocs,
    anomalies,
    stripBuffer: packed.stripBuffer,
    stripBufferEvent: packed.stripBufferEvent,
    eventTimes: packed.eventTimes,
    stripCount: packed.stripCount,
    maxBytesFull: (packed.maxBytesFull || raw.timeline.peak_bytes) * 1.1,
    segmentRows,
    traceEvents,
    flame,
    framePool,
    stackPool,
    stackByIdentity,
  };
  return { data };
}
