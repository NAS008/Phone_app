const DEFAULT_MAX_SIDE = 1400;
const DEFAULT_JPEG_QUALITY = 0.86;

class FileService {
  // ── Blob / bytes helpers ──────────────────────────────────────────────────

  async blobToBytes(blob) {
    if (!blob) return null;
    return new Uint8Array(await blob.arrayBuffer());
  }

  async dataUrlToBytes(dataUrl) {
    if (!dataUrl || typeof dataUrl !== "string" || !dataUrl.startsWith("data:")) {
      return null;
    }
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    return new Uint8Array(await blob.arrayBuffer());
  }

  // ── File / image loading ──────────────────────────────────────────────────

  fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Failed to read file"));
      reader.readAsDataURL(file);
    });
  }

  loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to load image"));
      img.src = src;
    });
  }

  // ── Canvas resize / encode ────────────────────────────────────────────────

  async prepareImageData(file, maxSide = DEFAULT_MAX_SIDE, quality = DEFAULT_JPEG_QUALITY) {
    const rawDataUrl = await this.fileToDataUrl(file);
    const image = await this.loadImage(rawDataUrl);
    let { width, height } = image;
    if (Math.max(width, height) > maxSide) {
      const ratio = maxSide / Math.max(width, height);
      width = Math.round(width * ratio);
      height = Math.round(height * ratio);
    }
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas is not available");
    ctx.drawImage(image, 0, 0, width, height);
    return canvas.toDataURL("image/jpeg", quality);
  }
}

const fileService = new FileService();
export default fileService;
