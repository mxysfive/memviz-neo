// Semantic coloring for timeline blocks.
//
// PyTorch's upstream MemoryViz uses d3.schemeTableau10, which reads much
// calmer on a dark UI than highly-saturated HSL colors. We keep that
// spirit here: stable Tableau-family colors, small per-instance variation,
// and a dark-background mix so large traces do not turn into neon noise.

const TABLEAU10: Array<[number, number, number]> = [
  [78, 121, 167],
  [242, 142, 43],
  [225, 87, 89],
  [118, 183, 178],
  [89, 161, 79],
  [237, 201, 72],
  [176, 122, 161],
  [255, 157, 167],
  [156, 117, 95],
  [186, 176, 172],
].map(([r, g, b]) => [r / 255, g / 255, b / 255] as [number, number, number]);

const BG: [number, number, number] = [0.04, 0.04, 0.045];
const PAPER: [number, number, number] = [0.93, 0.93, 0.90];

function hash32(n: number): number {
  let x = n | 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
}

function mix(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

/**
 * Stable color for one timeline block.
 *   hueKey      groups blocks with related meaning, usually top_frame_idx.
 *   instanceIdx spreads repeated allocations across nearby Tableau colors.
 */
export function blockColor(hueKey: number, instanceIdx: number): [number, number, number] {
  const h = hash32(hueKey);
  const baseIdx = (h + instanceIdx * 3) % TABLEAU10.length;
  const neighborIdx = (baseIdx + 1 + ((h >>> 8) % 3)) % TABLEAU10.length;
  const base = TABLEAU10[baseIdx];
  const neighbor = TABLEAU10[neighborIdx];

  // Keep repeated allocations distinguishable without full hue-wheel jumps.
  const familyShift = (instanceIdx % 4) * 0.08;
  const blended = mix(base, neighbor, familyShift);
  const softened = mix(blended, BG, 0.18);
  const lifted = mix(softened, PAPER, 0.06 + ((h >>> 16) % 3) * 0.025);

  return [
    clamp01(lifted[0]),
    clamp01(lifted[1]),
    clamp01(lifted[2]),
  ];
}
