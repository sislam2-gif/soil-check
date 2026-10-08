// predict.js — runs the trained field-vision model fully offline in the browser.
// Drop-in for LeafCheck's model/predict.js: same loadModel() / predict(img) contract.
//
// Needs ONNX Runtime Web loaded first (creates the global "ort"):
//   <script src="vendor/ort.wasm.min.js"></script>
// and these files next to ort.wasm.min.js in vendor/:
//   ort-wasm-simd-threaded.wasm, ort-wasm-simd-threaded.mjs
// model.onnx and labels.json sit in the same folder as this file.

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

// Same preprocessing as training (common.py center_square_resize + normalize).
function toTensor(img) {
  const size = meta.img;
  const w = img.naturalWidth || img.videoWidth || img.width;
  const h = img.naturalHeight || img.videoHeight || img.height;
  const side = Math.min(w, h) * meta.crop_frac;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, (w - side) / 2, (h - side) / 2, side, side, 0, 0, size, size);
  const px = ctx.getImageData(0, 0, size, size).data;
  const area = size * size;
  const data = new Float32Array(3 * area);
  const [m0, m1, m2] = meta.mean, [s0, s1, s2] = meta.std;
  for (let i = 0; i < area; i++) {
    data[i] = (px[i * 4] / 255 - m0) / s0;
    data[area + i] = (px[i * 4 + 1] / 255 - m1) / s1;
    data[2 * area + i] = (px[i * 4 + 2] / 255 - m2) / s2;
  }
  return new ort.Tensor("float32", data, [1, 3, size, size]);
}

// img: <img>, <canvas> or <video>. Returns
// { label, confidence, top3: [{label, confidence}], domain, id }
export async function predict(img) {
  if (!session) throw new Error("Call loadModel() first");
  const out = await session.run({ [inputName]: toTensor(img) });
  const probs = out[outputName].data;            // softmax already applied in the model
  const ranked = Array.from(probs, (p, i) => ({ i, p })).sort((a, b) => b.p - a.p);
  const top3 = ranked.slice(0, 3).map(({ i, p }) => ({
    label: meta.classes[i].label, confidence: p,
  }));
  const best = meta.classes[ranked[0].i];
  return { label: best.label, confidence: ranked[0].p, top3, domain: best.domain, id: best.id };
}
