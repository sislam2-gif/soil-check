// predict.js — runs the trained field-vision model fully offline in the browser.
// Drop-in for LeafCheck's model/predict.js: same loadModel() / predict(img) contract.
//
// Needs ONNX Runtime Web loaded first (creates the global "ort"):
//   <script src="vendor/ort.wasm.min.js"></script>
// and these files next to ort.wasm.min.js in vendor/:
//   ort-wasm-simd-threaded.wasm, ort-wasm-simd-threaded.mjs
// model.onnx and labels.json sit in the same folder as this file.
//
// Two ways to predict:
//   predict(img)       looks at the middle of the whole photo (plants, single leaves)
//   predictTiled(img)  cuts the photo into a grid of small squares, checks each, and
//                      combines them (soil texture: the model was trained on close-up patches)
// Python mirror of both, for testing on a PC: try_photos.py. Keep them in sync.

let session, meta, inputName, outputName;
const HERE = new URL(".", import.meta.url);

// wasmPaths: folder holding the ort .wasm/.mjs files. LeafCheck keeps them in vendor/,
// one level up from model/.
export async function loadModel({ wasmPaths = new URL("../vendor/", HERE).href } = {}) {
  meta = await (await fetch(new URL("labels.json", HERE))).json();
  ort.env.wasm.wasmPaths = wasmPaths;
  // Multi-threading needs cross-origin isolation headers, which GitHub Pages doesn't send.
  // One thread is plenty for a ~5M-parameter model.
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(new URL("model.onnx", HERE).href, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  inputName = session.inputNames[0];
  outputName = session.outputNames[0];
  // warm-up so the first real scan is fast
  const n = meta.img * meta.img * 3;
  await session.run({ [inputName]: new ort.Tensor("float32", new Float32Array(n), [1, 3, meta.img, meta.img]) });
}

export function getMeta() { return meta; }

const dims = img => [img.naturalWidth || img.videoWidth || img.width,
                     img.naturalHeight || img.videoHeight || img.height];

// Draw one square region of the image, resized to the model size, normalized, into data at offset.
// Big shrinks happen in halving steps: a single big shrink in the browser keeps more fine grain
// than training's resize did, which matters for textures like soil.
function drawRegion(img, sx, sy, side, data, offset) {
  const size = meta.img;
  let src = img, x = sx, y = sy, s = side;
  while (s > size * 2) {
    const half = Math.round(s / 2);
    const step = document.createElement("canvas");
    step.width = step.height = half;
    const sctx = step.getContext("2d");
    sctx.imageSmoothingEnabled = true;
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(src, x, y, s, s, 0, 0, half, half);
    src = step; x = 0; y = 0; s = half;
  }
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, x, y, s, s, 0, 0, size, size);
  const px = ctx.getImageData(0, 0, size, size).data;
  const area = size * size;
  const [m0, m1, m2] = meta.mean, [s0, s1, s2] = meta.std;
  for (let i = 0; i < area; i++) {
    data[offset + i] = (px[i * 4] / 255 - m0) / s0;
    data[offset + area + i] = (px[i * 4 + 1] / 255 - m1) / s1;
    data[offset + 2 * area + i] = (px[i * 4 + 2] / 255 - m2) / s2;
  }
}

// Run the model on a list of square regions [sx, sy, side] in one batch.
async function runRegions(img, regions) {
  const size = meta.img, per = 3 * size * size;
  const data = new Float32Array(per * regions.length);
  regions.forEach(([sx, sy, side], k) => drawRegion(img, sx, sy, side, data, k * per));
  const out = await session.run({ [inputName]: new ort.Tensor("float32", data, [regions.length, 3, size, size]) });
  const all = out[outputName].data, n = meta.classes.length;   // softmax already applied in the model
  return regions.map((_, k) => all.slice(k * n, (k + 1) * n));
}

function ranked(probs) {
  return Array.from(probs, (p, i) => ({ i, p })).sort((a, b) => b.p - a.p);
}

function result(probs, extra = {}) {
  const r = ranked(probs);
  const best = meta.classes[r[0].i];
  return {
    label: best.label, confidence: r[0].p, domain: best.domain, id: best.id,
    top3: r.slice(0, 3).map(({ i, p }) => ({ label: meta.classes[i].label, confidence: p })),
    ...extra,
  };
}

// img: <img>, <canvas> or <video>. Returns
// { label, confidence, top3: [{label, confidence}], domain, id }
export async function predict(img) {
  if (!session) throw new Error("Call loadModel() first");
  const [w, h] = dims(img);
  const side = Math.min(w, h) * meta.crop_frac;
  const [probs] = await runRegions(img, [[(w - side) / 2, (h - side) / 2, side]]);
  return result(probs);
}

// The largest centered square, split into g x g tiles, for each g in grids.
export function tileRegions(w, h, grids) {
  const side = Math.min(w, h), x0 = (w - side) / 2, y0 = (h - side) / 2;
  const regions = [];
  for (const g of grids) {
    const t = side / g;
    for (let r = 0; r < g; r++) for (let c = 0; c < g; c++) regions.push([x0 + c * t, y0 + r * t, t]);
  }
  return regions;
}

// Combine tile answers. Tiles whose top answer is an "other" (not relevant) class are
// ignored if enough tiles show the subject; the answer is the average over subject tiles.
export function combineTiles(tileProbs, minFrac) {
  const n = meta.classes.length;
  const isOther = meta.classes.map(c => c.domain === "other");
  const mean = rows => {
    const m = new Float32Array(n);
    for (const p of rows) for (let i = 0; i < n; i++) m[i] += p[i] / rows.length;
    return m;
  };
  const subject = tileProbs.filter(p => !isOther[ranked(p)[0].i]);
  const coverage = { tiles: tileProbs.length, matched: subject.length };
  if (!isOther.some(Boolean)) return result(mean(tileProbs), { coverage });
  if (subject.length && subject.length / tileProbs.length >= minFrac) {
    const m = mean(subject);
    for (let i = 0; i < n; i++) if (isOther[i]) m[i] = 0;
    return result(m, { coverage });
  }
  return result(mean(tileProbs), { coverage });
}

export async function predictTiled(img, grids = meta.tiles || [3, 4], minFrac = meta.tile_min_frac ?? 0.25) {
  if (!session) throw new Error("Call loadModel() first");
  const [w, h] = dims(img);
  return combineTiles(await runRegions(img, tileRegions(w, h, grids)), minFrac);
}
