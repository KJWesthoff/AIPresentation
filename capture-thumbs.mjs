import puppeteer from 'puppeteer';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FILE = 'file://' + path.join(__dirname, 'index.html');
const OUT  = path.join(__dirname, 'presentation_images', 'thumbs');
const W = 1280, H = 720;
const THUMB_W = 320, THUMB_H = 180;

const SLIDE_IDS = [
  'activation-functions','activation-functions-intro','agent-harness-overview',
  'agent-memory-1','agent-security','ai-agents','ai-agents-overview',
  'allyouneed','allyouneed_contd','attn-head','attn-masking',
  'backprop-network','backprop-simple','backprop-tensor','batch-epochs',
  'bert-architecture','bias-backprop','bias-variance','cls-token',
  'cnn-architecture','cnn-eval','cnn-fcn-anim','cnn-intro','cnn-params',
  'cnns','cnn-slabs-c1','cnn-slabs-c2','cnn-slabs-p1','cnn-slabs-p2',
  'cnn-training','conv-details','conv-filters','conv-full-digit',
  'cross-attention','cross-attn-diffusion','diy-modelling','early-ml',
  'embed-context','embeddings','enc-dec-concepts','encoder-finetuning',
  'encoder-only','ffn-activations','ffn-glu','ffn-layer','gen-llm-training',
  'gradient-landscape','human-annotation','llm-arch-families','llm-embeddings',
  'llm-params','lstm-animation','lstm-use-cases','matrix-equation',
  'mlp-architecture','mlp-backprop','mnist-nb-models','mnist-preprocessing',
  'mnist-to-vector','model-evaluation','moe','multi-head-attn','multimodal-ar',
  'nn-intro','no-more-mnist','optimization-algorithms','output-layer',
  'overfitting-epochs','paradigms','perceptron','phishing-usecase',
  'pos-encoding-rope','pos-encoding-sinusoidal','prompts-harnesses-agents',
  'qkv-attention','rnn-animation','rnn-limits','training-hyperparameters',
  'training-network','training-tips','transformer-3d-overview','unembedding',
  'vector-space-3d','vision-transformers','vit-demo','v-matrix',
  'webcam-demo','word-order-intro','word-vector-demo',
];

// slides that need extra time for animations to settle
const SLOW = new Set([
  'gradient-landscape','rnn-animation','lstm-animation','embed-context',
  'qkv-attention','v-matrix','attn-head','multi-head-attn',
  'pos-encoding-sinusoidal','pos-encoding-rope','ffn-layer','ffn-activations',
  'ffn-glu','transformer-3d-overview','moe','prompts-harnesses-agents',
  'cross-attention','cross-attn-diffusion','multimodal-ar','llm-arch-families',
  'gen-llm-training','human-annotation','encoder-only','encoder-finetuning',
  'cls-token','attn-masking','unembedding','bert-architecture','mnist-to-vector',
  'perceptron','output-layer','training-network',
]);

async function main() {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox','--disable-setuid-sandbox','--disable-web-security'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: W, height: H, deviceScaleFactor: 1 });

  console.log('Loading presentation…');
  await page.goto(FILE, { waitUntil: 'networkidle0', timeout: 30000 });

  // Wait for Reveal to be ready
  await page.waitForFunction(() => typeof Reveal !== 'undefined' && Reveal.isReady(), { timeout: 15000 });

  // Hide the slide-map chooser UI and any deck overlays while capturing
  await page.addStyleTag({ content: `
    #sm-chooser, #sm-tooltip, #sm-custom-hint, #sm-done-btn, #sm-reset-btn, #sm-sep { display:none !important; }
    .reveal .progress, .reveal .controls { display:none !important; }
  `});

  let done = 0, skipped = 0;
  for (const id of SLIDE_IDS) {
    const outFile = path.join(OUT, id + '.jpg');
    if (fs.existsSync(outFile)) { skipped++; continue; }

    // Navigate to the slide
    const found = await page.evaluate((slideId) => {
      const el = document.getElementById(slideId);
      if (!el) return false;
      const idx = Reveal.getIndices(el);
      Reveal.slide(idx.h, idx.v, idx.f);
      return true;
    }, id);

    if (!found) { console.warn(`  ! slide not found: ${id}`); continue; }

    const delay = SLOW.has(id) ? 1800 : 600;
    await new Promise(r => setTimeout(r, delay));

    const screenshot = await page.screenshot({ type: 'jpeg', quality: 82, fullPage: false });

    // Scale down with sharp if available, otherwise save as-is and note
    // We'll use Puppeteer's clip to emulate a smaller screenshot
    const scaled = await page.evaluate(
      ({ tw, th, sw, sh }) => {
        const canvas = document.createElement('canvas');
        canvas.width = tw; canvas.height = th;
        const ctx = canvas.getContext('2d');
        // We'll use a temporary image; actual scaling happens in Node below
        return { tw, th };
      },
      { tw: THUMB_W, th: THUMB_H, sw: W, sh: H }
    );

    // Save full-res then resize via Pillow via a small inline script —
    // actually just save JPEG at full res; we'll resize via CSS.
    // For true resize, write to tmp then call python.
    const tmpFile = outFile + '.tmp.jpg';
    fs.writeFileSync(tmpFile, screenshot);

    // Resize with Python/Pillow (already installed)
    const { execSync } = await import('child_process');
    execSync(`python3 -c "
from PIL import Image
img = Image.open('${tmpFile}')
img = img.resize((${THUMB_W}, ${THUMB_H}), Image.LANCZOS)
img.save('${outFile}', 'JPEG', quality=82, optimize=True)
"`);
    fs.unlinkSync(tmpFile);

    done++;
    process.stdout.write(`  [${done + skipped}/${SLIDE_IDS.length}] ${id}\n`);
  }

  await browser.close();
  console.log(`\nDone. ${done} captured, ${skipped} already existed.`);
}

main().catch(e => { console.error(e); process.exit(1); });
