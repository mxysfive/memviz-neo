import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { SegmentInfo } from "../types/snapshot";
import type { TimelineData } from "../types/timeline";
import type { TraceEvent } from "../compute";
import { blockColor } from "../compute/palette";
import { formatBytes, formatTopFrame } from "../utils";
import { useDataStore } from "../stores/dataStore";
import {
  COLOR_ACCENT,
  COLOR_AXIS,
  COLOR_BG,
  COLOR_DIVIDER,
  COLOR_LABEL_DIM,
  COLOR_PEAK,
  FONT_MONO,
  FONT_MONO_SM,
} from "./theme";

interface Props {
  data: TimelineData;
  segments: SegmentInfo[];
  traceEvents: TraceEvent[];
  width: number;
  height: number;
  currentRank: number;
}

interface StateSegment {
  addr: number;
  size: number;
  segmentType: string;
  stream: number;
}

interface StateBlock {
  addr: number;
  size: number;
  top_frame_idx: number;
  stack_idx: number;
  freeRequested: boolean;
  alloc_us: number;
}

interface SegmentLayout extends StateSegment {
  row: number;
  offset: number;
}

interface StateSnapshot {
  segments: StateSegment[];
  blocks: StateBlock[];
  reserved: number;
  allocated: number;
}

type Hover =
  | { kind: "block"; block: StateBlock; segment: SegmentLayout; mx: number; my: number }
  | { kind: "segment"; segment: SegmentLayout; mx: number; my: number }
  | null;

const MARGIN = { top: 18, right: 18, bottom: 24, left: 12 };
const TOOLBAR_H = 64;
const EVENT_ROW_H = 58;
const EVENT_OVERSCAN = 8;
const CHECKPOINT_STRIDE = 256;
const SEGMENT_GAP_PX = 24;

function eventTimeLabel(e: TraceEvent, data: TimelineData): string {
  return data.time_axis === "event_ordinal"
    ? `#${Math.round(e.time_us).toLocaleString()}`
    : `${((e.time_us - data.time_min) / 1e6).toFixed(6)}s`;
}

function eventKind(action: string): string {
  if (action.startsWith("segment_") || action === "segment_map" || action === "segment_unmap") return "segment";
  if (action.startsWith("free")) return "free";
  if (action === "alloc") return "alloc";
  if (action === "oom") return "oom";
  return "meta";
}

function formatEventTitle(e: TraceEvent): string {
  if (e.action === "snapshot") return "snapshot";
  return `${e.action} · ${formatBytes(e.size)}`;
}

function cloneSegments(input: Map<number, StateSegment>) {
  const out = new Map<number, StateSegment>();
  input.forEach((s, k) => out.set(k, { ...s }));
  return out;
}

function cloneBlocks(input: Map<number, StateBlock>) {
  const out = new Map<number, StateBlock>();
  input.forEach((b, k) => out.set(k, { ...b }));
  return out;
}

function makeEventBlock(e: TraceEvent, freeRequested: boolean): StateBlock {
  return {
    addr: e.addr,
    size: e.size,
    top_frame_idx: e.top_frame_idx,
    stack_idx: e.stack_idx,
    freeRequested,
    alloc_us: e.time_us,
  };
}

function makeEventSegment(e: TraceEvent): StateSegment {
  return {
    addr: e.addr,
    size: e.size,
    segmentType: e.action === "segment_map" ? "mapped" : "trace",
    stream: e.stream ?? 0,
  };
}

function segmentEnd(seg: StateSegment): number {
  return seg.addr + seg.size;
}

function canMergeSegments(a: StateSegment, b: StateSegment): boolean {
  return a.stream === b.stream;
}

function removeBlocksInRange(blockMap: Map<number, StateBlock>, start: number, end: number) {
  for (const b of Array.from(blockMap.values())) {
    if (b.addr >= start && b.addr < end) blockMap.delete(b.addr);
  }
}

function insertSegment(
  segmentMap: Map<number, StateSegment>,
  seg: StateSegment,
  merge: boolean,
) {
  if (seg.size <= 0) return;
  let next = { ...seg };
  if (merge) {
    let changed = true;
    while (changed) {
      changed = false;
      for (const existing of Array.from(segmentMap.values())) {
        if (!canMergeSegments(existing, next)) continue;
        if (segmentEnd(existing) === next.addr) {
          segmentMap.delete(existing.addr);
          next = {
            ...next,
            addr: existing.addr,
            size: existing.size + next.size,
            segmentType: existing.segmentType,
          };
          changed = true;
          break;
        }
        if (segmentEnd(next) === existing.addr) {
          segmentMap.delete(existing.addr);
          next = {
            ...next,
            size: next.size + existing.size,
            segmentType: next.segmentType === "mapped" ? existing.segmentType : next.segmentType,
          };
          changed = true;
          break;
        }
      }
    }
  }
  segmentMap.set(next.addr, next);
}

function removeSegmentRange(
  segmentMap: Map<number, StateSegment>,
  blockMap: Map<number, StateBlock>,
  seg: StateSegment,
  split: boolean,
) {
  if (seg.size <= 0) return;
  const start = seg.addr;
  const end = segmentEnd(seg);
  removeBlocksInRange(blockMap, start, end);

  if (!split) {
    segmentMap.delete(start);
    return;
  }

  const existing = Array.from(segmentMap.values()).find((s) => (
    s.addr <= start && end <= segmentEnd(s)
  ));
  if (!existing) {
    segmentMap.delete(start);
    return;
  }

  const existingEnd = segmentEnd(existing);
  segmentMap.delete(existing.addr);
  if (existing.addr < start) {
    segmentMap.set(existing.addr, {
      ...existing,
      size: start - existing.addr,
    });
  }
  if (end < existingEnd) {
    segmentMap.set(end, {
      ...existing,
      addr: end,
      size: existingEnd - end,
    });
  }
}

function unapplyEvent(
  segmentMap: Map<number, StateSegment>,
  blockMap: Map<number, StateBlock>,
  e: TraceEvent,
) {
  switch (e.action) {
    case "alloc":
      blockMap.delete(e.addr);
      break;
    case "free_requested": {
      const b = blockMap.get(e.addr);
      if (b) b.freeRequested = false;
      break;
    }
    case "free_completed":
      blockMap.set(e.addr, makeEventBlock(e, true));
      break;
    case "free":
      blockMap.set(e.addr, makeEventBlock(e, false));
      break;
    case "segment_alloc":
      removeSegmentRange(segmentMap, blockMap, makeEventSegment(e), false);
      break;
    case "segment_map":
      removeSegmentRange(segmentMap, blockMap, makeEventSegment(e), true);
      break;
    case "segment_free":
      insertSegment(segmentMap, makeEventSegment(e), false);
      break;
    case "segment_unmap":
      insertSegment(segmentMap, makeEventSegment(e), true);
      break;
  }
}

function applyEvent(
  segmentMap: Map<number, StateSegment>,
  blockMap: Map<number, StateBlock>,
  e: TraceEvent,
) {
  switch (e.action) {
    case "alloc":
      blockMap.set(e.addr, makeEventBlock(e, false));
      break;
    case "free_requested": {
      const b = blockMap.get(e.addr);
      if (b) b.freeRequested = true;
      break;
    }
    case "free_completed":
    case "free":
      blockMap.delete(e.addr);
      break;
    case "segment_alloc":
      insertSegment(segmentMap, makeEventSegment(e), false);
      break;
    case "segment_map":
      insertSegment(segmentMap, makeEventSegment(e), true);
      break;
    case "segment_free":
      removeSegmentRange(segmentMap, blockMap, makeEventSegment(e), false);
      break;
    case "segment_unmap":
      removeSegmentRange(segmentMap, blockMap, makeEventSegment(e), true);
      break;
  }
}

function buildFinalState(segments: SegmentInfo[]) {
  const segmentMap = new Map<number, StateSegment>();
  const blockMap = new Map<number, StateBlock>();
  for (const seg of segments) {
    segmentMap.set(seg.address, {
      addr: seg.address,
      size: seg.total_size,
      segmentType: seg.segment_type,
      stream: seg.stream ?? 0,
    });
    for (const b of seg.blocks) {
      if (b.state !== "active_allocated" && b.state !== "active_pending_free") continue;
      blockMap.set(b.address, {
        addr: b.address,
        size: b.size,
        top_frame_idx: b.top_frame_idx ?? -1,
        stack_idx: -1,
        freeRequested: b.state === "active_pending_free",
        alloc_us: -1,
      });
    }
  }
  return { segmentMap, blockMap };
}

interface ReplayCheckpoint {
  segmentMap: Map<number, StateSegment>;
  blockMap: Map<number, StateBlock>;
}

interface ReplayCache {
  events: TraceEvent[];
  checkpoints: Map<number, ReplayCheckpoint>;
  lastIdx: number;
}

function snapshotFromMaps(
  segmentMap: Map<number, StateSegment>,
  blockMap: Map<number, StateBlock>,
): StateSnapshot {
  const segments = [...segmentMap.values()].filter((s) => s.size > 0);
  const blocks = [...blockMap.values()].filter((b) => b.size > 0);
  return {
    segments,
    blocks,
    reserved: segments.reduce((sum, s) => sum + s.size, 0),
    allocated: blocks.reduce((sum, b) => sum + b.size, 0),
  };
}

function createReplayCache(
  finalSegments: Map<number, StateSegment>,
  finalBlocks: Map<number, StateBlock>,
  events: TraceEvent[],
): ReplayCache {
  const lastIdx = events.length - 1;
  const checkpoints = new Map<number, ReplayCheckpoint>();

  const initialSegments = cloneSegments(finalSegments);
  const initialBlocks = cloneBlocks(finalBlocks);
  for (let i = lastIdx; i >= 0; i--) {
    unapplyEvent(initialSegments, initialBlocks, events[i]);
  }

  checkpoints.set(-1, {
    segmentMap: initialSegments,
    blockMap: initialBlocks,
  });
  if (lastIdx >= 0) {
    checkpoints.set(lastIdx, {
      segmentMap: cloneSegments(finalSegments),
      blockMap: cloneBlocks(finalBlocks),
    });
  }

  return { events, checkpoints, lastIdx };
}

function replayStateCached(cache: ReplayCache, eventIdx: number): StateSnapshot {
  if (cache.lastIdx < 0) return snapshotFromMaps(new Map(), new Map());
  const targetIdx = Math.max(-1, Math.min(eventIdx, cache.lastIdx));

  let checkpointIdx = -1;
  for (const idx of cache.checkpoints.keys()) {
    if (idx <= targetIdx && idx > checkpointIdx) checkpointIdx = idx;
  }
  const checkpoint = cache.checkpoints.get(checkpointIdx) || cache.checkpoints.get(-1);
  if (!checkpoint) return snapshotFromMaps(new Map(), new Map());

  const segmentMap = cloneSegments(checkpoint.segmentMap);
  const blockMap = cloneBlocks(checkpoint.blockMap);
  for (let i = checkpointIdx + 1; i <= targetIdx; i++) {
    applyEvent(segmentMap, blockMap, cache.events[i]);
  }
  if (targetIdx - checkpointIdx >= CHECKPOINT_STRIDE) {
    cache.checkpoints.set(targetIdx, {
      segmentMap: cloneSegments(segmentMap),
      blockMap: cloneBlocks(blockMap),
    });
  }
  return snapshotFromMaps(segmentMap, blockMap);
}

function parseAddressQuery(raw: string): string {
  const q = raw.trim().toLowerCase();
  if (!q) return "";
  return q.startsWith("0x") ? q.slice(2) : q;
}

function eventMatchesAddress(e: TraceEvent, needle: string): boolean {
  if (!needle || e.addr === 0) return false;
  const hex = e.addr.toString(16).toLowerCase();
  const dec = String(e.addr);
  return hex.includes(needle) || dec.includes(needle);
}

function layoutSegments(segments: StateSegment[], plotW: number): {
  layout: SegmentLayout[];
  rows: number;
  maxRowSize: number;
} {
  const sorted = segments.slice().sort((a, b) => {
    if (a.size !== b.size) return a.size - b.size;
    return a.addr - b.addr;
  });
  const maxRowSize = Math.max(1, sorted[sorted.length - 1]?.size ?? 1);
  const pad = Math.max(1, maxRowSize * (SEGMENT_GAP_PX / Math.max(1, plotW)));
  let row = 0;
  let rowSize = 0;
  const layout: SegmentLayout[] = [];
  for (const seg of sorted) {
    if (rowSize > 0 && rowSize + seg.size > maxRowSize) {
      row++;
      rowSize = 0;
    }
    layout.push({ ...seg, row, offset: rowSize });
    rowSize += seg.size + pad;
  }
  return { layout, rows: row + 1, maxRowSize };
}

export default function AllocatorStateHistory({
  data,
  segments,
  traceEvents,
  width,
  height,
  currentRank,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const eventsRef = useRef<HTMLDivElement>(null);
  const [eventIdx, setEventIdx] = useState(() => Math.max(0, traceEvents.length - 1));
  const [eventScrollTop, setEventScrollTop] = useState(0);
  const [addressQuery, setAddressQuery] = useState("");
  const [hover, setHover] = useState<Hover>(null);
  const framePool = useDataStore((s) => s.framePool);
  const setSelectedAlloc = useDataStore((s) => s.setSelectedAlloc);
  const viewRef = useRef({ scale: 1, x: 0, y: 0 });
  const dragRef = useRef<{ x: number; y: number; ox: number; oy: number; moved: boolean } | null>(null);
  const suppressClickRef = useRef(false);

  useEffect(() => {
    setEventIdx(Math.max(0, traceEvents.length - 1));
    setEventScrollTop(0);
    setAddressQuery("");
    viewRef.current = { scale: 1, x: 0, y: 0 };
    setHover(null);
  }, [traceEvents, currentRank]);

  const finalState = useMemo(() => buildFinalState(segments), [segments]);
  const replayCache = useMemo(
    () => createReplayCache(finalState.segmentMap, finalState.blockMap, traceEvents),
    [finalState, traceEvents],
  );
  const safeIdx = traceEvents.length === 0
    ? -1
    : Math.max(0, Math.min(eventIdx, traceEvents.length - 1));
  const lastEventIdx = Math.max(0, traceEvents.length - 1);
  const state = useMemo(
    () => replayStateCached(replayCache, safeIdx),
    [replayCache, safeIdx],
  );

  const listW = Math.max(260, Math.min(420, Math.round(width * 0.34)));
  const rightW = Math.max(280, width - listW - 12);
  const canvasH = Math.max(180, height - TOOLBAR_H);
  const plotW = Math.max(80, rightW - MARGIN.left - MARGIN.right);
  const plotH = Math.max(80, canvasH - MARGIN.top - MARGIN.bottom);
  const segLayout = useMemo(
    () => layoutSegments(state.segments, plotW),
    [state.segments, plotW],
  );

  useEffect(() => {
    if (replayCache.lastIdx < 0) return;
    const initial = replayCache.checkpoints.get(-1);
    if (!initial) return;
    let cancelled = false;
    const segmentMap = cloneSegments(initial.segmentMap);
    const blockMap = cloneBlocks(initial.blockMap);
    let from = -1;
    let cursor = Math.min(replayCache.lastIdx, CHECKPOINT_STRIDE - 1);

    const pump = () => {
      if (cancelled) return;
      const deadline = performance.now() + 6;
      while (!cancelled && cursor <= replayCache.lastIdx && performance.now() < deadline) {
        for (; from < cursor; from++) {
          applyEvent(segmentMap, blockMap, traceEvents[from + 1]);
        }
        replayCache.checkpoints.set(cursor, {
          segmentMap: cloneSegments(segmentMap),
          blockMap: cloneBlocks(blockMap),
        });
        cursor += CHECKPOINT_STRIDE;
      }
      if (!cancelled && cursor <= replayCache.lastIdx) {
        window.setTimeout(pump, 0);
      }
    };

    const timer = window.setTimeout(pump, 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [replayCache, traceEvents]);

  useEffect(() => {
    const el = eventsRef.current;
    if (!el || safeIdx < 0) return;
    const top = safeIdx * EVENT_ROW_H;
    const bottom = top + EVENT_ROW_H;
    if (top < el.scrollTop) {
      el.scrollTop = top;
      setEventScrollTop(top);
    } else if (bottom > el.scrollTop + el.clientHeight) {
      const nextTop = Math.max(0, bottom - el.clientHeight);
      el.scrollTop = nextTop;
      setEventScrollTop(nextTop);
    }
  }, [safeIdx, canvasH]);

  const visibleStart = Math.max(0, Math.floor(eventScrollTop / EVENT_ROW_H) - EVENT_OVERSCAN);
  const visibleEnd = Math.min(
    traceEvents.length,
    Math.ceil((eventScrollTop + canvasH) / EVENT_ROW_H) + EVENT_OVERSCAN,
  );
  const visibleEvents = traceEvents.slice(visibleStart, visibleEnd);

  const searchNeedle = useMemo(() => parseAddressQuery(addressQuery), [addressQuery]);
  const searchMatches = useMemo(() => {
    if (!searchNeedle) return [];
    const matches: number[] = [];
    for (let i = 0; i < traceEvents.length; i++) {
      if (eventMatchesAddress(traceEvents[i], searchNeedle)) matches.push(i);
    }
    return matches;
  }, [traceEvents, searchNeedle]);
  const searchMatchSet = useMemo(() => new Set(searchMatches), [searchMatches]);
  const searchMatchIndex = useMemo(() => {
    const index = new Map<number, number>();
    searchMatches.forEach((eventIndex, ordinal) => index.set(eventIndex, ordinal));
    return index;
  }, [searchMatches]);
  const selectedMatchOrdinal = safeIdx >= 0 ? searchMatchIndex.get(safeIdx) ?? -1 : -1;

  const handleAddressQueryChange = useCallback((value: string) => {
    setAddressQuery(value);
    const needle = parseAddressQuery(value);
    if (!needle) return;
    for (let i = 0; i < traceEvents.length; i++) {
      if (eventMatchesAddress(traceEvents[i], needle)) {
        setEventIdx(i);
        return;
      }
    }
  }, [traceEvents]);

  const selectSearchMatch = useCallback((direction: 1 | -1) => {
    if (searchMatches.length === 0) return;
    const currentOrdinal = safeIdx >= 0 ? searchMatchIndex.get(safeIdx) ?? -1 : -1;
    if (currentOrdinal < 0) {
      if (direction > 0) {
        const firstForward = searchMatches.find((idx) => idx > safeIdx);
        setEventIdx(firstForward ?? searchMatches[0]);
      } else {
        let firstBackward: number | undefined;
        for (let i = searchMatches.length - 1; i >= 0; i--) {
          if (searchMatches[i] < safeIdx) {
            firstBackward = searchMatches[i];
            break;
          }
        }
        setEventIdx(firstBackward ?? searchMatches[searchMatches.length - 1]);
      }
      return;
    }
    const nextOrdinal = (currentOrdinal + direction + searchMatches.length) % searchMatches.length;
    setEventIdx(searchMatches[nextOrdinal]);
  }, [safeIdx, searchMatches, searchMatchIndex]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rightW * dpr;
    canvas.height = canvasH * dpr;
    canvas.style.width = `${rightW}px`;
    canvas.style.height = `${canvasH}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rightW, canvasH);
    ctx.fillStyle = COLOR_BG;
    ctx.fillRect(0, 0, rightW, canvasH);

    const { scale, x, y } = viewRef.current;
    const rowH = Math.max(9, Math.min(30, plotH / Math.max(1, segLayout.rows)));
    const baseX = plotW / Math.max(1, segLayout.maxRowSize);
    const baseY = rowH;
    const xScale = baseX * scale;
    const yScale = baseY * scale;
    const toX = (worldX: number) => MARGIN.left + x + worldX * xScale;
    const toY = (worldY: number) => MARGIN.top + y + worldY * yScale;
    const rowRectH = Math.max(2, yScale * 0.66);

    ctx.save();
    ctx.beginPath();
    ctx.rect(MARGIN.left, MARGIN.top, plotW, plotH);
    ctx.clip();

    // Segment base rectangles.
    for (const seg of segLayout.layout) {
      const sx = toX(seg.offset);
      const sy = toY(seg.row);
      const sw = seg.size * xScale;
      if (sx + sw < MARGIN.left || sx > MARGIN.left + plotW) continue;
      if (sy + rowRectH < MARGIN.top || sy > MARGIN.top + plotH) continue;
      ctx.fillStyle = "rgba(250,250,250,0.07)";
      ctx.fillRect(sx, sy, sw, rowRectH);
      ctx.strokeStyle = "rgba(250,250,250,0.42)";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(sx, sy, sw, rowRectH);
      if (sw > 72 && rowRectH > 12) {
        ctx.font = FONT_MONO_SM;
        ctx.fillStyle = "rgba(250,250,250,0.72)";
        ctx.fillText(`seg 0x${seg.addr.toString(16).slice(-6)} · ${formatBytes(seg.size)}`, sx + 4, sy + Math.min(rowRectH - 3, 12));
      }
    }

    const layoutByAddr = segLayout.layout.slice().sort((a, b) => a.addr - b.addr);
    function segmentForAddr(addr: number) {
      let lo = 0, hi = layoutByAddr.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const seg = layoutByAddr[mid];
        if (addr < seg.addr) hi = mid - 1;
        else if (addr >= seg.addr + seg.size) lo = mid + 1;
        else return seg;
      }
      return null;
    }

    for (const b of state.blocks) {
      const seg = segmentForAddr(b.addr);
      if (!seg) continue;
      const bx = toX(seg.offset + (b.addr - seg.addr));
      const by = toY(seg.row);
      const bw = Math.max(1, b.size * xScale);
      if (bx + bw < MARGIN.left || bx > MARGIN.left + plotW) continue;
      if (by + rowRectH < MARGIN.top || by > MARGIN.top + plotH) continue;
      const key = b.top_frame_idx >= 0 ? b.top_frame_idx : b.addr;
      const [r, g, bl] = blockColor(key, Math.abs(b.addr) % 7);
      ctx.fillStyle = b.freeRequested
        ? "rgba(248,113,113,0.88)"
        : `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(bl * 255)})`;
      ctx.fillRect(bx, by + 1, bw, Math.max(1, rowRectH - 2));
      if (bw > 96 && rowRectH > 14) {
        ctx.font = FONT_MONO_SM;
        ctx.fillStyle = "rgba(10,10,11,0.88)";
        const label = formatTopFrame(b.top_frame_idx, framePool) || `0x${b.addr.toString(16)}`;
        ctx.fillText(label.slice(0, Math.floor(bw / 6.5)), bx + 4, by + Math.min(rowRectH - 3, 12));
      }
    }
    ctx.restore();

    // Axes / frame.
    ctx.strokeStyle = COLOR_DIVIDER;
    ctx.strokeRect(MARGIN.left, MARGIN.top, plotW, plotH);
    ctx.fillStyle = COLOR_AXIS;
    ctx.font = FONT_MONO;
    ctx.textAlign = "left";
    const evt = safeIdx >= 0 ? traceEvents[safeIdx] : null;
    const eventLabel = evt
      ? `${evt.action} · ${formatBytes(evt.size)} · 0x${evt.addr.toString(16)}`
      : "No trace events";
    ctx.fillText(eventLabel, MARGIN.left, canvasH - 7);
    ctx.textAlign = "right";
    ctx.fillStyle = COLOR_LABEL_DIM;
    ctx.fillText(`${segLayout.layout.length} segments`, rightW - MARGIN.right, canvasH - 7);

    if (state.allocated > data.peak_bytes * 0.9 && data.peak_bytes > 0) {
      ctx.fillStyle = COLOR_PEAK;
      ctx.fillRect(MARGIN.left, 0, Math.min(plotW, (state.allocated / data.peak_bytes) * plotW), 2);
    }
  }, [rightW, canvasH, plotW, plotH, segLayout, state, framePool, safeIdx, traceEvents, data.peak_bytes]);

  useEffect(() => { draw(); }, [draw, hover]);

  const hitTest = useCallback((mx: number, my: number): Hover => {
    if (mx < MARGIN.left || mx > MARGIN.left + plotW || my < MARGIN.top || my > MARGIN.top + plotH) {
      return null;
    }
    const { scale, x, y } = viewRef.current;
    const rowH = Math.max(9, Math.min(30, plotH / Math.max(1, segLayout.rows)));
    const baseX = plotW / Math.max(1, segLayout.maxRowSize);
    const xScale = baseX * scale;
    const yScale = rowH * scale;
    const worldX = (mx - MARGIN.left - x) / xScale;
    const worldY = (my - MARGIN.top - y) / yScale;
    const row = Math.floor(worldY);
    const segment = segLayout.layout.find((s) => (
      s.row === row && worldX >= s.offset && worldX <= s.offset + s.size
    ));
    if (!segment) return null;
    const addr = segment.addr + (worldX - segment.offset);
    let best: StateBlock | null = null;
    for (const b of state.blocks) {
      if (b.addr >= segment.addr && b.addr < segment.addr + segment.size) {
        if (addr >= b.addr && addr <= b.addr + b.size) {
          if (!best || b.size < best.size) best = b;
        }
      }
    }
    if (best) return { kind: "block", block: best, segment, mx, my };
    return { kind: "segment", segment, mx, my };
  }, [plotW, plotH, segLayout, state.blocks]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    if (dragRef.current) {
      if (Math.abs(mx - dragRef.current.x) > 2 || Math.abs(my - dragRef.current.y) > 2) {
        dragRef.current.moved = true;
      }
      viewRef.current.x = dragRef.current.ox + (mx - dragRef.current.x);
      viewRef.current.y = dragRef.current.oy + (my - dragRef.current.y);
      draw();
      return;
    }
    setHover(hitTest(mx, my));
  }, [draw, hitTest]);

  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const view = viewRef.current;
    const prevScale = view.scale;
    const nextScale = Math.max(0.5, Math.min(80, prevScale * Math.exp(-e.deltaY * 0.0015)));
    if (nextScale === prevScale) return;
    const px = mx - MARGIN.left - view.x;
    const py = my - MARGIN.top - view.y;
    const ratio = nextScale / prevScale;
    view.x = mx - MARGIN.left - px * ratio;
    view.y = my - MARGIN.top - py * ratio;
    view.scale = nextScale;
    draw();
  }, [draw]);

  const event = safeIdx >= 0 ? traceEvents[safeIdx] : null;
  const eventTime = event
    ? eventTimeLabel(event, data)
    : "no events";
  const eventFrame = event
    ? formatTopFrame(event.top_frame_idx, framePool) || (event.addr ? `0x${event.addr.toString(16)}` : "")
    : "";

  return (
    <div
      className="state-history"
      style={{ width, height }}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "ArrowDown") {
          setEventIdx((i) => Math.min(traceEvents.length - 1, i + 1));
          e.preventDefault();
        } else if (e.key === "ArrowUp") {
          setEventIdx((i) => Math.max(0, i - 1));
          e.preventDefault();
        } else if (e.key === "Home") {
          setEventIdx(0);
          e.preventDefault();
        } else if (e.key === "End") {
          setEventIdx(Math.max(0, traceEvents.length - 1));
          e.preventDefault();
        }
      }}
    >
      <div className="state-toolbar mono">
        <div className="state-toolbar-left">
          <span className="hl">R{String(currentRank).padStart(2, "0")}</span>
          <span className="faint">Allocator State</span>
          <span>{safeIdx >= 0 ? `${safeIdx + 1}/${traceEvents.length}` : "0/0"}</span>
        </div>
        <div className="state-controls">
          <div className="state-search">
            <input
              type="search"
              value={addressQuery}
              placeholder="addr 0x..."
              spellCheck={false}
              onChange={(e) => handleAddressQueryChange(e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  selectSearchMatch(e.shiftKey ? -1 : 1);
                  e.preventDefault();
                }
              }}
            />
            <button
              type="button"
              disabled={searchMatches.length === 0}
              onClick={() => selectSearchMatch(-1)}
            >
              Prev hit
            </button>
            <button
              type="button"
              disabled={searchMatches.length === 0}
              onClick={() => selectSearchMatch(1)}
            >
              Next hit
            </button>
            <span className="state-search-count">
              {searchNeedle ? `${selectedMatchOrdinal >= 0 ? selectedMatchOrdinal + 1 : 0}/${searchMatches.length}` : ""}
            </span>
          </div>
          <div className="state-range-controls">
            <button type="button" onClick={() => setEventIdx((i) => Math.max(0, i - 1))}>Prev</button>
            <input
              type="range"
              min={0}
              max={Math.max(0, traceEvents.length - 1)}
              value={Math.max(0, safeIdx)}
              onChange={(e) => setEventIdx(Number(e.currentTarget.value))}
            />
            <button type="button" onClick={() => setEventIdx((i) => Math.min(lastEventIdx, i + 1))}>Next</button>
            <button
              type="button"
              onClick={() => {
                setEventIdx(lastEventIdx);
                viewRef.current = { scale: 1, x: 0, y: 0 };
                setHover(null);
              }}
            >
              Latest
            </button>
          </div>
        </div>
        <div className="state-toolbar-right">
          <span>{state.segments.length} segments</span>
          <span>{formatBytes(state.allocated)} allocated</span>
          <span className="faint">/ {formatBytes(state.reserved)} reserved</span>
        </div>
      </div>
      <div className="state-split" style={{ height: canvasH }}>
        <div
          ref={eventsRef}
          className="state-events"
          style={{ width: listW }}
          onScroll={(e) => setEventScrollTop(e.currentTarget.scrollTop)}
        >
          <div
            className="state-events-spacer"
            style={{ height: traceEvents.length * EVENT_ROW_H }}
          >
            {visibleEvents.map((e, localIdx) => {
              const i = visibleStart + localIdx;
              const selected = i === safeIdx;
              const matched = searchMatchSet.has(i);
              const frame = formatTopFrame(e.top_frame_idx, framePool);
              return (
                <button
                  key={`${i}-${e.action}-${e.addr}-${e.time_us}`}
                  type="button"
                  data-event-idx={i}
                  className={`state-event-row is-${eventKind(e.action)}${selected ? " is-selected" : ""}${matched ? " is-match" : ""}`}
                  style={{ height: EVENT_ROW_H, transform: `translateY(${i * EVENT_ROW_H}px)` }}
                  onClick={() => setEventIdx(i)}
                >
                  <span className="state-event-index">{i.toLocaleString()}</span>
                  <span className="state-event-main">
                    <span className="state-event-title">{formatEventTitle(e)}</span>
                    <span className="state-event-sub">
                      {eventTimeLabel(e, data)}
                      {e.addr ? ` · 0x${e.addr.toString(16)}` : ""}
                    </span>
                    {frame && <span className="state-event-frame">{frame}</span>}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
        <div className="state-canvas-panel" style={{ width: rightW, height: canvasH }}>
          <div className="state-current-event mono">
            <span className={`state-current-pill is-${event ? eventKind(event.action) : "meta"}`}>
              {event ? event.action : "none"}
            </span>
            <span>{eventTime}</span>
            {event && event.addr !== 0 && <span>0x{event.addr.toString(16)}</span>}
            {event && event.size > 0 && <span>{formatBytes(event.size)}</span>}
            {eventFrame && <span className="faint">{eventFrame}</span>}
          </div>
          <canvas
            ref={canvasRef}
            className="tl-canvas"
            style={{ cursor: dragRef.current ? "grabbing" : "grab" }}
            onMouseDown={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              dragRef.current = {
                x: e.clientX - rect.left,
                y: e.clientY - rect.top,
                ox: viewRef.current.x,
                oy: viewRef.current.y,
                moved: false,
              };
            }}
            onMouseMove={handleMouseMove}
            onMouseUp={() => {
              suppressClickRef.current = !!dragRef.current?.moved;
              dragRef.current = null;
            }}
            onMouseLeave={() => { dragRef.current = null; setHover(null); }}
            onDoubleClick={() => { viewRef.current = { scale: 1, x: 0, y: 0 }; draw(); }}
            onWheel={handleWheel}
            onClick={() => {
              if (suppressClickRef.current) {
                suppressClickRef.current = false;
                return;
              }
              if (hover?.kind === "block" && hover.block.alloc_us >= 0) {
                setSelectedAlloc({ addr: hover.block.addr, alloc_us: hover.block.alloc_us });
              }
            }}
          />
          {hover && (
            <StateHoverCard
              hover={hover}
              framePool={framePool}
              containerW={rightW}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function StateHoverCard({
  hover,
  framePool,
  containerW,
}: {
  hover: Exclude<Hover, null>;
  framePool: import("../types/snapshot").FrameRecord[];
  containerW: number;
}) {
  const CARD_W = 310;
  const PAD = 12;
  const left = hover.mx + PAD + CARD_W > containerW ? Math.max(4, hover.mx - PAD - CARD_W) : hover.mx + PAD;
  const top = Math.max(4, hover.my + PAD);
  const isBlock = hover.kind === "block";
  const title = isBlock
    ? formatBytes(hover.block.size)
    : formatBytes(hover.segment.size);
  const frame = isBlock
    ? formatTopFrame(hover.block.top_frame_idx, framePool) || `0x${hover.block.addr.toString(16)}`
    : hover.segment.segmentType;
  return (
    <div
      className="tl-hover-card mono"
      style={{
        position: "absolute",
        left,
        top,
        width: CARD_W,
        pointerEvents: "none",
        borderLeft: `2px solid ${isBlock && hover.block.freeRequested ? COLOR_PEAK : COLOR_ACCENT}`,
        display: "block",
      }}
    >
      <div className="eyebrow">
        {isBlock ? (hover.block.freeRequested ? "Pending Free Block" : "Allocated Block") : "Segment"}
      </div>
      <div style={{ color: "var(--fg)", fontSize: 14, marginTop: 2 }}>{title}</div>
      <div style={{ color: "var(--fg-muted)", fontSize: 11, marginTop: 2 }}>{frame}</div>
      <div className="faint" style={{ fontSize: 10, marginTop: 2 }}>
        {isBlock
          ? `0x${hover.block.addr.toString(16)} · +${formatBytes(hover.block.addr - hover.segment.addr)} in segment`
          : `0x${hover.segment.addr.toString(16)}`}
      </div>
    </div>
  );
}
