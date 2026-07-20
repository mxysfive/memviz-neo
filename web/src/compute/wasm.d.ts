declare module "../../../wasm/pkg/memviz_wasm.js" {
  export default function init(options?: { module: WebAssembly.Module }): void;
  export function initSync(options: { module: WebAssembly.Module }): void;
  export function parse_intern(data: Uint8Array, rank: number, layout_limit: number): string;
  export function parse_intern_binary(data: Uint8Array, rank: number): any;
}

declare module "../../../wasm/pkg/memviz_wasm_bg.wasm?url" {
  const url: string;
  export default url;
}
