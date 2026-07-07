// Semantic coloring for timeline blocks.
//
// The base hue is still derived from the block's "top frame" (first
// meaningful frame in the call stack), but repeated allocations from one
// site deliberately jump around the color wheel instead of only changing
// lightness. Training traces often execute one op hundreds of times in a
// row; if those blocks all stay in one hue family, adjacent blocks become
// hard to distinguish.
//
// Worker-friendly: no DOM / GL imports.

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; }
  else if (h < 120) { r = x; g = c; }
  else if (h < 180) { g = c; b = x; }
  else if (h < 240) { g = x; b = c; }
  else if (h < 300) { r = x; b = c; }
  else { r = c; b = x; }
  return [r + m, g + m, b + m];
}

const PHI  = 0.61803398875;
const PHI2 = 0.41421356237;

function hash01(n: number): number {
  let x = n | 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return (x >>> 0) / 0x100000000;
}

/**
 * Stable color for one timeline block.
 *   hueKey     — groups blocks that share meaning (usually top_frame_idx;
 *                callers fall back to stack_idx / addr when no frame).
 *   instanceIdx — 0-based counter among blocks sharing the same hueKey.
 *                 Drives high-contrast hue jumps for repeated allocs
 *                 from the same line/operator.
 */
export function blockColor(hueKey: number, instanceIdx: number): [number, number, number] {
  const base = hash01(hueKey);
  const hue = ((base + instanceIdx * PHI) % 1) * 360;
  const lig = 0.48 + hash01(hueKey ^ Math.imul(instanceIdx + 1, 0x9e3779b1)) * 0.22;
  const sat = 0.62 + ((instanceIdx * PHI2 + base) % 1) * 0.24;
  return hslToRgb(hue, sat, lig);
}
