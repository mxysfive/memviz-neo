import { create } from "zustand";
import {
  createWorkerPool,
  type WorkerPool,
  type WorkerTask,
  type LoadPhase,
  type ProgressSnapshot,
  type RankSummary as WorkerRankSummary,
} from "../compute/workerPool";
import { setSummary as cacheSetSummary, clearSummaries } from "./rankStore";

type SnapshotReader = () => Promise<ArrayBuffer>;
type SnapshotEntry = {
  name: string;
  file?: File;
  reader?: SnapshotReader;
  size?: number;
};

// Active pool lives until the next dataset is loaded or reset is called.
// Kept around so rank-switch requestFull() can talk to the worker that
// holds the target rank's data.
let activePool: WorkerPool | null = null;

export function getActivePool(): WorkerPool | null {
  return activePool;
}

export async function dumpRankIR(rank: number, allocationLimit: number | null = 1000): Promise<unknown> {
  if (!activePool) throw new Error("No active snapshot pool");
  return activePool.requestDebugDump(rank, { allocationLimit });
}

export async function dumpRankIRText(rank: number, allocationLimit: number | null = 1000): Promise<string> {
  return JSON.stringify(await dumpRankIR(rank, allocationLimit), null, 2);
}

export async function downloadRankIRDump(rank: number, allocationLimit: number | null = 1000): Promise<void> {
  const text = await dumpRankIRText(rank, allocationLimit);
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `memviz-rank${rank}-ir-dump.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

import { persistentNumber } from "../utils";

const HW_CONC = typeof navigator !== "undefined" ? (navigator.hardwareConcurrency || 4) : 4;
export const WORKER_COUNT_MAX = Math.max(4, Math.min(HW_CONC, 16));

const workerCountPref = persistentNumber(
  "memviz.workerCount",
  Math.min(HW_CONC, 8),
  (n) => n >= 1 && n <= WORKER_COUNT_MAX,
);

// Number of allocation events laid out and rendered per rank. The
// parser keeps all allocation records; changing this value replays the
// current rank from the cached IR without reading the pickle again.
// 0 = display all allocations.
export const LAYOUT_LIMIT_OPTIONS: { value: number; label: string }[] = [
  { value: 3000, label: "3k" },
  { value: 10000, label: "10k" },
  { value: 20000, label: "20k" },
  { value: 0, label: "all" },
];

const layoutLimitPref = persistentNumber(
  "memviz.layoutLimit",
  20000,
  (n) => n >= 0,
);

export function getLayoutLimit(): number {
  return useFileStore.getState().layoutLimit;
}

interface FileState {
  status: "idle" | "loading" | "ready" | "error";
  fileNames: string[];
  progress: number;
  phase: LoadPhase | "idle";
  completedCount: number;
  inFlightCount: number;
  totalCount: number;
  inFlightRanks: number[];
  poolSize: number;
  bytesLoaded: number;
  bytesTotal: number;
  activeMs: number;
  workerCount: number;
  layoutLimit: number;
  error: string | null;
  ranks: number[];

  openDirectory: () => Promise<void>;
  openFiles: (files: FileList) => Promise<void>;
  setWorkerCount: (n: number) => void;
  setLayoutLimit: (n: number) => void;
  reset: () => void;
}

function extractRank(filename: string): number {
  const m = filename.match(/rank(\d+)/);
  return m ? parseInt(m[1], 10) : 0;
}

export const useFileStore = create<FileState>((set) => ({
  status: "idle",
  fileNames: [],
  progress: 0,
  phase: "idle",
  completedCount: 0,
  inFlightCount: 0,
  totalCount: 0,
  inFlightRanks: [],
  poolSize: 0,
  bytesLoaded: 0,
  bytesTotal: 0,
  activeMs: 0,
  workerCount: workerCountPref.load(),
  layoutLimit: layoutLimitPref.load(),
  error: null,
  ranks: [],

  openDirectory: async () => {
    try {
      const dirHandle = await (window as any).showDirectoryPicker();
      const entries: SnapshotEntry[] = [];
      const stack: any[] = [dirHandle];
      while (stack.length > 0) {
        const dir = stack.pop();
        for await (const entry of dir.values()) {
          if (entry.kind === "directory") {
            stack.push(entry);
          } else if (entry.kind === "file" && entry.name.endsWith(".pickle")) {
            const handle = entry;
            const file = await handle.getFile();
            entries.push({ name: entry.name, file, size: file.size });
          }
        }
      }
      await loadAllParallel(entries, set);
    } catch (e: any) {
      if (e.name === "AbortError") return;
      set({ status: "error", error: String(e) });
    }
  },

  openFiles: async (fileList: FileList) => {
    const entries = Array.from(fileList)
      .filter((f) => f.name.endsWith(".pickle"))
      .map((f) => ({ name: f.name, file: f, size: f.size }));
    await loadAllParallel(entries, set);
  },

  setWorkerCount: (n: number) => {
    const clamped = Math.max(1, Math.min(n, WORKER_COUNT_MAX));
    workerCountPref.save(clamped);
    set({ workerCount: clamped });
  },

  setLayoutLimit: (n: number) => {
    const v = Math.max(0, Math.floor(n));
    layoutLimitPref.save(v);
    set({ layoutLimit: v });
  },

  reset: () => {
    if (activePool) {
      activePool.terminate();
      activePool = null;
    }
    clearSummaries();
    set({
      status: "idle",
      fileNames: [],
      progress: 0,
      phase: "idle",
      completedCount: 0,
      inFlightCount: 0,
      totalCount: 0,
      inFlightRanks: [],
      poolSize: 0,
      bytesLoaded: 0,
      bytesTotal: 0,
      activeMs: 0,
      error: null,
      ranks: [],
    });
  },
}));

if (import.meta.env.DEV && typeof window !== "undefined") {
  (window as any).__memvizLoadUrls = async (urls: string[]) => {
    const entries = await Promise.all(
      urls.map(async (url) => {
        const name = url.split("/").pop() || url;
        const buf = await (await fetch(url)).arrayBuffer();
        return { name, reader: async () => buf, size: buf.byteLength };
      }),
    );
    const set = (partial: Partial<FileState>) => useFileStore.setState(partial);
    await loadAllParallel(entries, set);
  };
}

if (typeof window !== "undefined") {
  (window as any).__memvizDumpRank = dumpRankIR;
  (window as any).__memvizDumpRankText = dumpRankIRText;
  (window as any).__memvizDownloadRankDump = downloadRankIRDump;
}

async function loadAllParallel(
  entries: SnapshotEntry[],
  set: (partial: Partial<FileState>) => void,
) {
  if (entries.length === 0) { set({ status: "error", error: "No .pickle files found" }); return; }

  const items = entries
    .map((e) => ({
      rank: extractRank(e.name),
      name: e.name,
      file: e.file,
      reader: e.reader,
      size: e.size ?? e.file?.size,
    }))
    .sort((a, b) => a.rank - b.rank);

  clearSummaries();
  set({
    status: "loading",
    fileNames: items.map((i) => i.name),
    progress: 0,
    phase: "compile_wasm",
    completedCount: 0,
    inFlightCount: 0,
    totalCount: items.length,
    inFlightRanks: [],
    poolSize: 0,
    bytesLoaded: 0,
    bytesTotal: items.reduce((sum, i) => sum + (i.size ?? 0), 0),
    activeMs: 0,
    error: null,
    ranks: items.map((i) => i.rank),
  });

  const tasks: WorkerTask[] = items.map((i) => ({
    rank: i.rank,
    file: i.file,
    getBuffer: i.reader,
    size: i.size,
  }));

  let firstDone = false;

  if (activePool) activePool.terminate();

  const desiredWorkers = useFileStore.getState().workerCount;
  const pool = createWorkerPool(
    (rank: number, summary: WorkerRankSummary) => {
      // Summary-only push during load: ~64 bytes per rank. Cheap
      // structured clone, cheap selector comparisons on main.
      cacheSetSummary(rank, summary);
      if (!firstDone) {
        firstDone = true;
        set({ status: "ready" });
      }
    },
    (rank, error) => {
      console.error(`[memviz] rank ${rank} failed:`, error);
    },
    (snap: ProgressSnapshot) => {
      set({
        progress: snap.progress,
        phase: snap.phase,
        completedCount: snap.completed,
        inFlightCount: snap.inFlight,
        totalCount: snap.total,
        inFlightRanks: snap.inFlightRanks,
        poolSize: snap.poolSize,
        bytesLoaded: snap.bytesLoaded,
        bytesTotal: snap.bytesTotal,
        activeMs: snap.activeMs,
      });
    },
    { poolSize: desiredWorkers },
  );
  activePool = pool;

  await pool.processAll(tasks);

  if (!firstDone) {
    set({ status: "error", error: "All ranks failed to parse", progress: 1 });
  } else {
    set({ progress: 1, bytesLoaded: useFileStore.getState().bytesTotal });
  }
}
