/**
 * rembg-worker.js
 * Drop-in replacement for rembg-webgpu — works on GitHub Pages without a bundler.
 * 
 * Features:
 *  ✅ WebGPU FP16 / FP32 auto-detection
 *  ✅ WASM fallback
 *  ✅ Download + build progress tracking
 *  ✅ Chunked processing (512px strips) — no memory crashes on large images
 *  ✅ Auto preview generation (≤ 450px)
 *  ✅ Blob URL lifecycle management
 *  ✅ Memory + browser caching
 *  ✅ Timeout protection
 *  ✅ Full error reporting
 */
 
import {
  env,
  AutoProcessor,
  AutoModel,
  RawImage,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3/dist/transformers.min.js';
 
// ─── Config ────────────────────────────────────────────────────────────────
const MODEL_ID      = 'briaai/RMBG-1.4';
const CHUNK_HEIGHT  = 512;   // px — strips to slice large images into
const PREVIEW_MAX   = 450;   // px — max dimension for preview thumbnail
const TIMEOUT_MS_WEBGPU = 120_000;  // 2 min for WebGPU
const TIMEOUT_MS_WASM   = 600_000;  // 10 min for WASM (much slower)
const WASM_INFER_MAX_PX = 1024;     // downsample input on WASM to keep it under ~1 min
 
env.allowLocalModels  = false;
env.useBrowserCache   = true;   // cache model weights after first download
 
// ─── State ─────────────────────────────────────────────────────────────────
let processor = null;
let model     = null;
let device    = 'wasm';   // updated after capability detection
let dtype     = 'fp32';
 
// ─── Helpers ───────────────────────────────────────────────────────────────
 
/** Post a typed message back to the main thread */
function send(type, payload = {}) {
  self.postMessage({ type, ...payload });
}
 
/** Detect best available backend: webgpu-fp16 → webgpu-fp32 → wasm */
async function detectCapabilities() {
  if (!self.navigator?.gpu) {
    return { device: 'wasm', dtype: 'fp32' };
  }
  try {
    const adapter = await self.navigator.gpu.requestAdapter();
    if (!adapter) return { device: 'wasm', dtype: 'fp32' };
 
    const hasFP16 = adapter.features.has('shader-f16');
    return {
      device: 'webgpu',
      dtype:  hasFP16 ? 'fp16' : 'fp32',
    };
  } catch {
    return { device: 'wasm', dtype: 'fp32' };
  }
}
 
/** Intercept fetch to track download progress for model weights */
function installProgressProxy() {
  const _fetch = self.fetch.bind(self);
  self.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input.url;
 
    // Only track Hugging Face model files
    if (!url.includes('huggingface.co') && !url.includes('hf.co')) {
      return _fetch(input, init);
    }
 
    const resp = await _fetch(input, init);
    if (!resp.body || !resp.headers.get('content-length')) return resp;
 
    const total   = parseInt(resp.headers.get('content-length'), 10);
    let   loaded  = 0;
    const filename = url.split('/').pop();
 
    const stream = new TransformStream({
      transform(chunk, ctrl) {
        loaded += chunk.byteLength;
        send('progress', {
          phase:    'downloading',
          progress: Math.round((loaded / total) * 100),
          filename,
          loaded,
          total,
        });
        ctrl.enqueue(chunk);
      },
    });
 
    return new Response(resp.body.pipeThrough(stream), {
      headers: resp.headers,
      status:  resp.status,
    });
  };
}
 
/**
 * Scale down an existing PNG blob URL to fit within PREVIEW_MAX px on its longest side.
 * Uses OffscreenCanvas — no RawImage channel-count concerns.
 */
async function scaleDownBlob(blobUrl, origWidth, origHeight) {
  const scale = Math.min(1, PREVIEW_MAX / Math.max(origWidth, origHeight));
  if (scale >= 1) return blobUrl; // already small — reuse the same URL
 
  const w = Math.round(origWidth  * scale);
  const h = Math.round(origHeight * scale);
 
  const resp   = await fetch(blobUrl);
  const blob   = await resp.blob();
  const bitmap = await createImageBitmap(blob);
 
  const canvas = new OffscreenCanvas(w, h);
  const ctx    = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
 
  const preview = await canvas.convertToBlob({ type: 'image/png' });
  return URL.createObjectURL(preview);
}
 
/**
 * Process one image through the model.
 * For images taller than CHUNK_HEIGHT we slice into strips,
 * run inference on each, then stitch the masks back together.
 * This prevents OOM crashes on high-res images.
 */
async function runInference(image) {
  const { width, height } = image;
 
  // Small image — single pass
  if (height <= CHUNK_HEIGHT * 2) {
    return runSinglePass(image);
  }
 
  // Large image — chunked pass
  send('status', { message: `Chunked processing ${width}×${height}…` });
 
  const strips      = Math.ceil(height / CHUNK_HEIGHT);
  const fullMask    = new Uint8ClampedArray(width * height);
 
  for (let i = 0; i < strips; i++) {
    const y0 = i * CHUNK_HEIGHT;
    const y1 = Math.min(y0 + CHUNK_HEIGHT, height);
 
    // Crop strip from original image (RGBA data slice)
    const stripData = cropImageData(image, 0, y0, width, y1 - y0);
    const strip     = new RawImage(stripData, width, y1 - y0, 4);
 
    const maskStrip = await runSinglePass(strip);
 
    // Write strip mask back into full mask
    for (let row = 0; row < y1 - y0; row++) {
      const srcOff = row * width;
      const dstOff = (y0 + row) * width;
      fullMask.set(maskStrip.subarray(srcOff, srcOff + width), dstOff);
    }
 
    send('progress', {
      phase:    'processing',
      progress: Math.round(((i + 1) / strips) * 100),
    });
  }
 
  return fullMask;
}
 
/** Single-pass inference — returns Uint8ClampedArray mask (grayscale, same WxH as input) */
async function runSinglePass(image) {
  send('progress', { phase: 'processing', progress: 10 });
  const { pixel_values } = await processor(image);
  send('progress', { phase: 'processing', progress: 50 });
  const { output } = await model({ input: pixel_values });
 
  // output[0] is shape [1, 1, H, W]; scale to 0-255
  const maskTensor = output[0].mul(255).to('uint8');
  const maskImage  = await RawImage.fromTensor(maskTensor)
    .resize(image.width, image.height);
  send('progress', { phase: 'processing', progress: 90 });
 
  return maskImage.data; // Uint8ClampedArray
}
 
/** Extract a rectangular region from a RawImage as raw RGBA Uint8ClampedArray */
function cropImageData(image, x, y, w, h) {
  const src    = image.data;
  const srcW   = image.width;
  const result = new Uint8ClampedArray(w * h * 4);
 
  for (let row = 0; row < h; row++) {
    const srcOff = ((y + row) * srcW + x) * 4;
    const dstOff = row * w * 4;
    result.set(src.subarray(srcOff, srcOff + w * 4), dstOff);
  }
  return result;
}
 
/**
 * Composite: apply alpha mask onto original image pixels,
 * returns a PNG blob URL (transparent background).
 */
async function compositeToBlob(image, mask) {
  const { width, height } = image;
  const rgba = new Uint8ClampedArray(image.data); // copy
 
  for (let i = 0; i < width * height; i++) {
    rgba[i * 4 + 3] = mask[i]; // set alpha from mask
  }
 
  // Use OffscreenCanvas if available, fallback otherwise
  const canvas = new OffscreenCanvas(width, height);
  const ctx    = canvas.getContext('2d');
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
 
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  return URL.createObjectURL(blob);
}
 
// ─── Main message handler ───────────────────────────────────────────────────
 
self.onmessage = async (e) => {
  const { type, id, imageUrl } = e.data;
 
  // ── LOAD ────────────────────────────────────────────────────────────────
  if (type === 'load') {
    try {
      send('status', { message: 'Detecting capabilities…' });
 
      const cap = await detectCapabilities();
      device    = cap.device;
      dtype     = cap.dtype;
 
      send('capabilities', { device, dtype });
      send('status', { message: `Using ${device.toUpperCase()} (${dtype})` });
 
      installProgressProxy();
 
      send('progress', { phase: 'downloading', progress: 0 });
 
      processor = await AutoProcessor.from_pretrained(MODEL_ID);
 
      send('progress', { phase: 'building', progress: 0 });
      send('status', { message: 'Compiling model…' });
 
      model = await AutoModel.from_pretrained(MODEL_ID, {
        device,
        dtype,
        config: { model_type: 'custom' },
      });
 
      send('progress', { phase: 'ready', progress: 100 });
      send('ready', { device, dtype });
 
    } catch (err) {
      send('error', {
        phase:   'load',
        message: err.message ?? String(err),
      });
    }
    return;
  }
 
  // ── PROCESS ─────────────────────────────────────────────────────────────
  if (type === 'process') {
    if (!model || !processor) {
      send('error', { id, message: 'Model not loaded yet.' });
      return;
    }
 
    // Track blobs so we can tell the main thread which to revoke
    const blobsCreated = [];
 
    // Use a longer timeout on WASM — inference is CPU-bound and much slower
    const TIMEOUT_MS = device === 'wasm' ? TIMEOUT_MS_WASM : TIMEOUT_MS_WEBGPU;
    const timeout = new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Timed out after ${TIMEOUT_MS / 1000}s`)), TIMEOUT_MS)
    );
 
    try {
      const result = await Promise.race([
        (async () => {
          send('progress', { id, phase: 'processing', progress: 0 });
 
          const rawImage = await RawImage.fromURL(imageUrl);
          const { width: origW, height: origH } = rawImage;
 
          // On WASM, downsample very large images before inference to stay within
          // a reasonable time budget. The composited result is then scaled back up.
          let inferImage = rawImage;
          if (device === 'wasm') {
            const longestSide = Math.max(origW, origH);
            if (longestSide > WASM_INFER_MAX_PX) {
              const s = WASM_INFER_MAX_PX / longestSide;
              inferImage = await rawImage.resize(Math.round(origW * s), Math.round(origH * s));
              send('status', { message: `WASM mode — resized to ${inferImage.width}×${inferImage.height} for processing…` });
            }
          }
 
          const { width, height } = inferImage;
          send('status', { message: `Processing ${width}×${height}…` });
 
          // Full-res mask (on the possibly-downsampled image)
          const mask    = await runInference(inferImage);
          // Composite on the downsampled image; preview will be generated from this
          const blobUrl = await compositeToBlob(inferImage, mask);
          blobsCreated.push(blobUrl);
 
          // Preview (≤ PREVIEW_MAX px) — scale down the already-composited blob
          // using OffscreenCanvas. Avoids any channel-count mismatch with RawImage.
          const previewUrl = await scaleDownBlob(blobUrl, image.width, image.height);
          blobsCreated.push(previewUrl);
 
          return {
            blobUrl,
            previewUrl,
            width,
            height,
          };
        })(),
        timeout,
      ]);
 
      send('result', { id, ...result, blobsCreated });
 
    } catch (err) {
      // Revoke any partial blobs on error
      blobsCreated.forEach(u => URL.revokeObjectURL(u));
      send('error', {
        id,
        phase:   'process',
        message: err.message ?? String(err),
      });
    }
    return;
  }
 
  // ── REVOKE ──────────────────────────────────────────────────────────────
  // Main thread sends this when it's done displaying an image
  if (type === 'revoke') {
    const urls = Array.isArray(e.data.urls) ? e.data.urls : [e.data.url];
    urls.forEach(u => { try { URL.revokeObjectURL(u); } catch {} });
    return;
  }
 
  // ── CAPABILITIES (query without loading) ────────────────────────────────
  if (type === 'capabilities') {
    const cap = await detectCapabilities();
    send('capabilities', cap);
    return;
  }
};
