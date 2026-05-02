/* D3.js Neural Network Animations — MNIST illustration
   Slides: 1) MNIST→Vector  2) Output Layer  3) Perceptron  4) Full Network
*/
(function () {
  'use strict';

  const C = {
    primary:     '#00d9ff',
    secondary:   '#9d4edd',
    accent:      '#ff006e',
    text:        '#f8f9fa',
    muted:       '#adb5bd',
    bg:          '#1a1a2e',
    conn:        'rgba(0,217,255,0.10)',
    connHi:      'rgba(0,217,255,0.85)',
    nodeStroke:  'rgba(0,217,255,0.6)',
    nodeFill:    'rgba(0,217,255,0.12)',
    nodeActive:  'rgba(0,217,255,0.85)',
  };

  // ── MNIST sample — injected by mnist-sample.js (window.MNIST_SAMPLE) ──────
  // Using a JS global avoids fetch() being blocked on file:// protocol.
  const _sample    = window.MNIST_SAMPLE || { pixels: new Array(784).fill(0), label: 0, index: 0 };
  const PIXEL_DATA = _sample.pixels;
  const MNIST_LABEL = _sample.label;

  // Compatibility shim: callers do loadSample().then(...) — resolve immediately
  function loadSample() { return Promise.resolve(); }

  // ── Shared input pixel samples — used by both MNIST→Vector and Full Network ──
  // Uses the same center-row window as the MNIST→Vector slide (VEC_START logic):
  // find the row near row 14 with the most ink, then pick 10 evenly-spaced pixels
  // from a 22-pixel window.  Values and order therefore match slide 2's vector column.
  const INPUT_SAMPLES = (() => {
    const MID_ROW = 14, WINDOW = 22, SHOW = 10;
    let bestStart = MID_ROW * 28, bestSum = -1;
    for (let r = MID_ROW - 5; r <= MID_ROW + 5; r++) {
      const start = r * 28;
      const sum = PIXEL_DATA.slice(start, start + WINDOW).reduce((a, v) => a + v, 0);
      if (sum > bestSum) { bestSum = sum; bestStart = start; }
    }
    return Array.from({length: WINDOW}, (_, k) => {
      const idx = bestStart + k;
      return { v: PIXEL_DATA[idx], idx };
    });
  })();

  // ── Helper: arrow marker defs ────────────────────────────────────────────
  function addArrowDef(svg, id, color) {
    svg.append('defs').append('marker')
      .attr('id', id)
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 5).attr('markerHeight', 5)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,0 L10,5 L0,10 Z')
      .attr('fill', color || C.primary);
  }

  // ── SLIDE 1: MNIST Grid → Input Vector ──────────────────────────────────
  function initMNISTToVector(el) {
    loadSample().then(() => _renderMNISTToVector(el));
  }
  function _renderMNISTToVector(el) {
    el.innerHTML = '';
    const W = 1100, H = 520;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%').style('overflow', 'visible');

    addArrowDef(svg, 'arr-primary', C.primary);

    const CELL = 13;
    const GAP  = 1.5;           // gap between pixels — eliminates aliasing
    const GW = 28 * CELL;
    const GH = 28 * CELL;
    const GX = 30;
    const GY = (H - GH) / 2;

    // ── Grid cells ──
    PIXEL_DATA.forEach((val, idx) => {
      const row = Math.floor(idx / 28), col = idx % 28;
      const g = Math.round(val * 255);
      svg.append('rect')
        .attr('x', GX + col * CELL + GAP).attr('y', GY + row * CELL + GAP)
        .attr('width', CELL - GAP * 2).attr('height', CELL - GAP * 2)
        .attr('fill', `rgb(${g},${g},${g})`)
        .attr('opacity', 0)
        .transition().delay(idx * 0.3).duration(250)
        .attr('opacity', 1);
    });

    // Grid label
    svg.append('text').attr('x', GX + GW / 2).attr('y', GY - 12)
      .attr('text-anchor', 'middle').attr('fill', C.primary)
      .attr('font-size', 13).attr('font-family', 'inherit')
      .text(`28 × 28 grayscale image  (label: ${MNIST_LABEL})`);

    svg.append('text').attr('x', GX + GW / 2).attr('y', GY + GH + 18)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 11).attr('font-family', 'monospace')
      .attr('opacity', 0).text('pixel ∈ [0, 1]  — MNIST test set')
      .transition().delay(500).duration(400).attr('opacity', 1);

    // Pick sample pixels spread across the brightness range [0, 0.25, 0.5, 0.75+]
    // so the annotated values look meaningfully different from each other
    const buckets = [
      { min: 0.00, max: 0.12 },
      { min: 0.20, max: 0.45 },
      { min: 0.50, max: 0.72 },
      { min: 0.78, max: 1.00 },
    ];
    const SAMPLES = buckets.map(({ min, max }) => {
      // collect candidates inside the bucket, prefer pixels not on the border rows/cols
      const candidates = PIXEL_DATA
        .map((v, i) => ({ v, i }))
        .filter(({ v, i }) => {
          const r = Math.floor(i / 28), c = i % 28;
          return v >= min && v <= max && r > 1 && r < 26 && c > 1 && c < 26;
        });
      if (!candidates.length) return null;
      // pick middle candidate for determinism
      return candidates[Math.floor(candidates.length / 2)].i;
    }).filter(i => i !== null);

    SAMPLES.forEach((idx, i) => {
      const row = Math.floor(idx / 28), col = idx % 28;
      const val = PIXEL_DATA[idx];
      svg.append('rect')
        .attr('x', GX + col * CELL + GAP - 1.5).attr('y', GY + row * CELL + GAP - 1.5)
        .attr('width', CELL - GAP * 2 + 3).attr('height', CELL - GAP * 2 + 3)
        .attr('fill', 'none').attr('stroke', C.accent).attr('stroke-width', 1.5)
        .attr('opacity', 0)
        .transition().delay(600 + i * 80).duration(200).attr('opacity', 1);
      svg.append('text')
        .attr('x', GX + col * CELL + CELL).attr('y', GY + row * CELL + CELL - 3)
        .attr('fill', C.accent).attr('font-size', 9).attr('font-family', 'monospace')
        .attr('opacity', 0).text(val.toFixed(2))
        .transition().delay(600 + i * 80).duration(200).attr('opacity', 1);
    });

    // Flatten arrow
    const AX1 = GX + GW + 18, AY = H / 2;
    const arrowG = svg.append('g').attr('opacity', 0);
    arrowG.append('line')
      .attr('x1', AX1).attr('y1', AY).attr('x2', AX1 + 55).attr('y2', AY)
      .attr('stroke', C.primary).attr('stroke-width', 2)
      .attr('marker-end', 'url(#arr-primary)');
    arrowG.append('text').attr('x', AX1 + 27).attr('y', AY - 9)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 12).text('flatten');
    arrowG.transition().delay(700).duration(400).attr('opacity', 1);

    // ── Vector column ──
    const VX = AX1 + 72;
    const BOX_H = 19, BOX_W = 52, VEC_GAP = 2;
    const SHOW = 22;
    const totalH = SHOW * (BOX_H + VEC_GAP);
    const VY0 = (H - totalH) / 2;

    // Find a window of SHOW pixels near the centre of the image with real content.
    // Scan row-by-row around the vertical midpoint and pick the row whose
    // segment has the highest pixel sum (most ink).
    const MID_ROW = 14;  // start searching near row 14 (of 28)
    let bestStart = MID_ROW * 28, bestSum = -1;
    for (let r = MID_ROW - 5; r <= MID_ROW + 5; r++) {
      const start = r * 28;
      const sum = PIXEL_DATA.slice(start, start + SHOW).reduce((a, v) => a + v, 0);
      if (sum > bestSum) { bestSum = sum; bestStart = start; }
    }
    const VEC_START = bestStart;

    for (let i = 0; i < SHOW; i++) {
      const val = PIXEL_DATA[VEC_START + i];
      const g = Math.round(val * 255);
      const vy = VY0 + i * (BOX_H + VEC_GAP);
      svg.append('rect')
        .attr('x', VX).attr('y', vy)
        .attr('width', BOX_W).attr('height', BOX_H)
        .attr('fill', `rgb(${g},${g},${g})`)
        .attr('stroke', 'rgba(0,217,255,0.25)').attr('stroke-width', 0.5)
        .attr('opacity', 0)
        .transition().delay(900 + i * 20).duration(200).attr('opacity', 1);
      svg.append('text')
        .attr('x', VX + BOX_W + 6).attr('y', vy + BOX_H - 4)
        .attr('fill', val > 0.25 ? C.text : C.muted)
        .attr('font-size', 9).attr('font-family', 'monospace')
        .attr('opacity', 0).text(val.toFixed(2))
        .transition().delay(900 + i * 20).duration(200).attr('opacity', 1);
    }

    // Ellipsis
    const EY = VY0 + SHOW * (BOX_H + VEC_GAP) + 8;
    // ⋮ above — showing we're mid-vector
    svg.append('text').attr('x', VX + BOX_W / 2).attr('y', VY0 - 10)
      .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 14)
      .attr('opacity', 0).text('⋮')
      .transition().delay(880).duration(300).attr('opacity', 1);

    // index label for first shown entry
    svg.append('text').attr('x', VX - 6).attr('y', VY0 + BOX_H - 4)
      .attr('text-anchor', 'end').attr('fill', C.muted).attr('font-size', 8)
      .attr('font-family', 'monospace').attr('opacity', 0)
      .text(`x[${VEC_START}]`)
      .transition().delay(1000).duration(300).attr('opacity', 1);

    // ⋮ below
    svg.append('text').attr('x', VX + BOX_W / 2).attr('y', EY)
      .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 14)
      .attr('opacity', 0).text('⋮')
      .transition().delay(1350).duration(300).attr('opacity', 1);

    // Bracket
    const BX = VX - 12, BY0b = VY0 - 6, BY1b = EY + 28;
    svg.append('path')
      .attr('d', `M${BX+8},${BY0b} L${BX},${BY0b} L${BX},${BY1b} L${BX+8},${BY1b}`)
      .attr('fill', 'none').attr('stroke', C.primary).attr('stroke-width', 1.5)
      .attr('opacity', 0)
      .transition().delay(850).duration(400).attr('opacity', 0.55);

    // Labels
    svg.append('text').attr('x', VX + BOX_W / 2).attr('y', EY + 26)
      .attr('text-anchor', 'middle').attr('fill', C.primary)
      .attr('font-size', 11).attr('font-weight', 'bold').attr('opacity', 0)
      .text('784 values')
      .transition().delay(1450).duration(300).attr('opacity', 1);

    svg.append('text').attr('x', VX + BOX_W / 2).attr('y', H - 8)
      .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 12)
      .attr('font-style', 'italic').attr('opacity', 0)
      .text('x  ∈  ℝ⁷⁸⁴   (input vector)')
      .transition().delay(1550).duration(400).attr('opacity', 1);

    // ── Styled perceptron-format vector ───────────────────────────────────
    const VX2 = 648, R2 = 20, SP2 = 48;

    // Second arrow: pixel vector → styled vector
    svg.append('line')
      .attr('x1', VX + BOX_W + 42).attr('y1', H / 2)
      .attr('x2', VX2 - R2 - 6).attr('y2', H / 2)
      .attr('stroke', C.primary).attr('stroke-width', 2)
      .attr('marker-end', 'url(#arr-primary)')
      .attr('opacity', 0)
      .transition().delay(1650).duration(400).attr('opacity', 1);

    // Entries: use actual pixel values from the VEC_START window
    const sub = n => String(n).split('').map(d => '₀₁₂₃₄₅₆₇₈₉'[+d]).join('');
    const STYLED = [
      { label: 'x' + sub(1),   val: PIXEL_DATA[VEC_START] },
      { label: 'x' + sub(2),   val: PIXEL_DATA[VEC_START + 1] },
      { label: 'x' + sub(3),   val: PIXEL_DATA[VEC_START + 2] },
      { label: 'x' + sub(4),   val: PIXEL_DATA[VEC_START + 3] },
      { label: '⋮',             val: null },
      { label: 'x₇₈₄',         val: PIXEL_DATA[783] },
    ];
    const N2 = STYLED.length;
    const VY0_2 = (H - (N2 - 1) * SP2) / 2;  // y of first circle centre

    // Header label above the styled vector
    svg.append('text').attr('x', VX2).attr('y', VY0_2 - R2 - 18)
      .attr('text-anchor', 'middle').attr('fill', C.primary)
      .attr('font-size', 11).attr('font-weight', 'bold').attr('font-family', 'monospace')
      .attr('opacity', 0).text('x ∈ ℝ⁷⁸⁴')
      .transition().delay(2080).duration(400).attr('opacity', 1);

    STYLED.forEach(({ label, val }, i) => {
      const cy = VY0_2 + i * SP2;
      const delay = 1720 + i * 70;

      if (label === '⋮') {
        svg.append('text').attr('x', VX2).attr('y', cy + 6)
          .attr('text-anchor', 'middle').attr('fill', C.muted)
          .attr('font-size', 18).attr('font-family', 'inherit')
          .attr('opacity', 0).text('⋮')
          .transition().delay(delay).duration(300).attr('opacity', 1);
        return;
      }
      svg.append('circle').attr('cx', VX2).attr('cy', cy).attr('r', R2)
        .attr('fill', C.nodeFill).attr('stroke', C.nodeStroke).attr('stroke-width', 1.5)
        .attr('opacity', 0)
        .transition().delay(delay).duration(300).attr('opacity', 1);
      svg.append('text').attr('x', VX2).attr('y', cy - 4)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 11).attr('font-family', 'monospace')
        .attr('opacity', 0).text(label)
        .transition().delay(delay).duration(300).attr('opacity', 1);
      if (val !== null) {
        svg.append('text').attr('x', VX2).attr('y', cy + 9)
          .attr('text-anchor', 'middle').attr('fill', C.muted)
          .attr('font-size', 9).attr('font-family', 'monospace')
          .attr('opacity', 0).text(val.toFixed(2))
          .transition().delay(delay + 100).duration(300).attr('opacity', 1);
      }
    });

    // Left bracket
    const BX2  = VX2 - R2 - 12;
    const BY0_2 = VY0_2 - R2 - 6;
    const BY1_2 = VY0_2 + (N2 - 1) * SP2 + R2 + 6;
    svg.append('path')
      .attr('d', `M${BX2 + 8},${BY0_2} L${BX2},${BY0_2} L${BX2},${BY1_2} L${BX2 + 8},${BY1_2}`)
      .attr('fill', 'none').attr('stroke', C.primary).attr('stroke-width', 1.5)
      .attr('opacity', 0)
      .transition().delay(1680).duration(400).attr('opacity', 0.6);

    // Right bracket
    const RBX2 = VX2 + R2 + 4;
    svg.append('path')
      .attr('d', `M${RBX2},${BY0_2} L${RBX2 + 8},${BY0_2} L${RBX2 + 8},${BY1_2} L${RBX2},${BY1_2}`)
      .attr('fill', 'none').attr('stroke', C.primary).attr('stroke-width', 1.5)
      .attr('opacity', 0)
      .transition().delay(1680).duration(400).attr('opacity', 0.6);

    // Right-side summary box
    const SX = 726, SY = 60, SW = 220;
    const sumG = svg.append('g').attr('opacity', 0);
    sumG.append('rect')
      .attr('x', SX).attr('y', SY).attr('width', SW).attr('height', 200)
      .attr('fill', 'rgba(0,217,255,0.06)').attr('stroke', 'rgba(0,217,255,0.3)')
      .attr('stroke-width', 1).attr('rx', 8);
    sumG.append('text').attr('x', SX + SW / 2).attr('y', SY + 24)
      .attr('text-anchor', 'middle').attr('fill', C.primary)
      .attr('font-size', 13).attr('font-weight', 'bold').text('Input Layer');
    const lines = [
      '28 × 28 = 784 pixels',
      'Each pixel → float [0, 1]',
      'Flattened to 1-D vector',
      'x ∈ ℝ⁷⁸⁴',
    ];
    lines.forEach((t, i) => {
      sumG.append('text').attr('x', SX + 14).attr('y', SY + 56 + i * 28)
        .attr('fill', C.text).attr('font-size', 12).text(t);
    });
    sumG.transition().delay(1600).duration(500).attr('opacity', 1);
  }

  // ── SLIDE 2: Output Layer (10 nodes 0-9) ────────────────────────────────
  function initOutputLayer(el) {
    loadSample().then(() => _renderOutputLayer(el));
  }
  function _renderOutputLayer(el) {
    el.innerHTML = '';
    const W = 1100, H = 500;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%');

    addArrowDef(svg, 'arr2', C.primary);

    // ── Left panel: MNIST pixel grid ─────────────────────────────────────────
    const CELL = 8, GAP = 1;
    const GW = 28 * CELL, GH = 28 * CELL;
    const GX = 12, GY = (H - GH) / 2;

    PIXEL_DATA.forEach((val, idx) => {
      const row = Math.floor(idx / 28), col = idx % 28;
      const g = Math.round(val * 255);
      svg.append('rect')
        .attr('x', GX + col * CELL + GAP).attr('y', GY + row * CELL + GAP)
        .attr('width', CELL - GAP * 2).attr('height', CELL - GAP * 2)
        .attr('fill', `rgb(${g},${g},${g})`)
        .attr('opacity', 0)
        .transition().delay(idx * 0.15).duration(200)
        .attr('opacity', 1);
    });

    svg.append('text').attr('x', GX + GW / 2).attr('y', GY - 10)
      .attr('text-anchor', 'middle').attr('fill', C.primary)
      .attr('font-size', 12)
      .text(`28 × 28  (label: ${MNIST_LABEL})`);

    // Arrow from grid to output panel
    const ARR_X1 = GX + GW + 10, ARR_X2 = GX + GW + 42, ARR_Y = H / 2;
    svg.append('line')
      .attr('x1', ARR_X1).attr('y1', ARR_Y).attr('x2', ARR_X2).attr('y2', ARR_Y)
      .attr('stroke', C.primary).attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arr2)')
      .attr('opacity', 0)
      .transition().delay(500).duration(300).attr('opacity', 0.6);

    // ── Right panel: output nodes + probability bars ──────────────────────────
    const probs     = _sample.probs;
    const predicted = _sample.predicted;
    const R = 24;
    const SPACING = H / 10;
    const NY = (i) => SPACING / 2 + i * SPACING;
    const NX = GX + GW + 62;
    const BAR_X = NX + R + 16;
    const MAX_BAR = 340;

    svg.append('text').attr('x', NX).attr('y', 20)
      .attr('text-anchor', 'middle').attr('fill', C.primary)
      .attr('font-size', 13).attr('font-weight', 'bold')
      .text('Output Layer — 10 classes');

    for (let i = 0; i < 10; i++) {
      const cy = NY(i);
      const isMax = i === predicted;
      const col = isMax ? C.accent : C.primary;

      svg.append('circle')
        .attr('cx', NX).attr('cy', cy).attr('r', R)
        .attr('fill', isMax ? 'rgba(255,0,110,0.18)' : C.nodeFill)
        .attr('stroke', col).attr('stroke-width', isMax ? 2 : 1)
        .attr('opacity', 0)
        .transition().delay(i * 80).duration(350).attr('opacity', 1);

      svg.append('text').attr('x', NX).attr('y', cy + 5)
        .attr('text-anchor', 'middle').attr('fill', col)
        .attr('font-size', 15).attr('font-weight', isMax ? 'bold' : 'normal')
        .attr('opacity', 0).text(String(i))
        .transition().delay(i * 80).duration(350).attr('opacity', 1);

      svg.append('rect')
        .attr('x', BAR_X).attr('y', cy - 10)
        .attr('width', MAX_BAR).attr('height', 20)
        .attr('fill', 'rgba(255,255,255,0.04)').attr('rx', 3)
        .attr('opacity', 0)
        .transition().delay(i * 80 + 200).duration(300).attr('opacity', 1);

      const barW = Math.pow(probs[i], 0.3) * MAX_BAR;
      svg.append('rect')
        .attr('x', BAR_X).attr('y', cy - 10)
        .attr('width', 0).attr('height', 20)
        .attr('fill', isMax ? C.accent : 'rgba(0,217,255,0.45)').attr('rx', 3)
        .transition().delay(i * 80 + 400).duration(600)
        .attr('width', barW);

      const p = probs[i];
      const probLabel = p >= 0.0001
        ? (p * 100).toFixed(p >= 0.01 ? 1 : 3) + '%'
        : p.toExponential(1);
      svg.append('text')
        .attr('x', BAR_X + barW + 5).attr('y', cy + 5)
        .attr('fill', isMax ? C.accent : C.muted)
        .attr('font-size', 11).attr('font-family', 'monospace')
        .attr('font-weight', isMax ? 'bold' : 'normal')
        .attr('opacity', 0).text(probLabel)
        .transition().delay(i * 80 + 700).duration(300).attr('opacity', 1);
    }

    // "Predicted: N" badge
    const badgeX = BAR_X + MAX_BAR + 18;
    const badgeG = svg.append('g').attr('opacity', 0);
    badgeG.append('rect')
      .attr('x', badgeX).attr('y', NY(predicted) - 38)
      .attr('width', 150).attr('height', 76)
      .attr('fill', 'rgba(255,0,110,0.1)').attr('stroke', C.accent)
      .attr('stroke-width', 2).attr('rx', 10);
    badgeG.append('text')
      .attr('x', badgeX + 75).attr('y', NY(predicted) - 14)
      .attr('text-anchor', 'middle').attr('fill', C.accent)
      .attr('font-size', 12).text('Predicted');
    badgeG.append('text')
      .attr('x', badgeX + 75).attr('y', NY(predicted) + 24)
      .attr('text-anchor', 'middle').attr('fill', C.accent)
      .attr('font-size', 30).attr('font-weight', 'bold').text(String(predicted));
    badgeG.transition().delay(1200).duration(500).attr('opacity', 1);

    svg.append('text').attr('x', BAR_X + MAX_BAR / 2).attr('y', H - 6)
      .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 11)
      .attr('font-style', 'italic').attr('opacity', 0)
      .text('real MLP predictions (784→64→32→10, 92.6% test accuracy) — bar scale: p⁰·³')
      .transition().delay(1400).duration(400).attr('opacity', 1);
  }

  // ── SLIDE 3: Perceptron ──────────────────────────────────────────────────
  function initPerceptron(el) {
    el.innerHTML = '';
    const W = 1100, H = 500;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%');

    addArrowDef(svg, 'arr3', C.primary);

    // ── Data ─────────────────────────────────────────────────────────────
    const INPUTS = [
      { label: 'x₁', val: 0.84 },
      { label: 'x₂', val: 0.12 },
      { label: 'x₃', val: 0.91 },
      { label: 'x₄', val: 0.03 },
      { label: '⋮',  val: null },
      { label: 'xₙ', val: 0.67 },
    ];
    const WV = [ 0.52, -0.73,  0.31, 0.88, null, 0.45];
    const WL = ['w₁',  'w₂', 'w₃', 'w₄',  '',  'wₙ'];
    const BIAS = 0.15;

    // Only real (non-ellipsis) inputs
    const active = INPUTS
      .map((inp, i) => ({ ...inp, wi: WV[i], wl: WL[i], iy: i }))
      .filter(d => d.val !== null);
    active.forEach(d => { d.prod = Math.round(d.wi * d.val * 10000) / 10000; });
    const wsum  = Math.round(active.reduce((s, d) => s + d.prod, 0) * 10000) / 10000;
    const z     = Math.round((wsum + BIAS) * 10000) / 10000;
    const sigm  = v => Math.round((1 / (1 + Math.exp(-v))) * 10000) / 10000;
    const aVal  = sigm(z);

    // ── Layout ────────────────────────────────────────────────────────────
    const N  = INPUTS.length;
    const IX = 75;
    const IY = i => 50 + i * (H - 75) / (N - 1);
    const NX = 400, NY = H / 2, NR = 46;
    const OX = 560;
    const BIAS_X = NX, BIAS_Y = NY + 128;
    const PX = 660;   // panel left edge  (660 → 1085 = 425 px wide)

    // Panel column x-coords
    const CwEnd = PX + 115;  // right-align weight "wᵢ=±val"
    const Cmul  = PX + 128;  // × centre
    const CxEnd = PX + 198;  // right-align input "xᵢ=val"
    const Ceq   = PX + 211;  // = centre
    const CpEnd = PX + 273;  // right-align product
    const Csep  = PX + 288;  // vertical divider
    const CaccV = PX + 413;  // right-align running accumulator

    // ── Input nodes ───────────────────────────────────────────────────────
    INPUTS.forEach(({ label, val }, i) => {
      if (label === '⋮') {
        svg.append('text').attr('x', IX).attr('y', IY(i) + 6)
          .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 18)
          .attr('opacity', 0).text('⋮')
          .transition().delay(i * 80).duration(300).attr('opacity', 1);
        return;
      }
      svg.append('circle').attr('cx', IX).attr('cy', IY(i)).attr('r', 22)
        .attr('fill', C.nodeFill).attr('stroke', C.nodeStroke).attr('stroke-width', 1.5)
        .attr('opacity', 0).transition().delay(i * 80).duration(300).attr('opacity', 1);
      svg.append('text').attr('x', IX).attr('y', IY(i) - 5)
        .attr('text-anchor', 'middle').attr('fill', C.primary).attr('font-size', 13)
        .attr('opacity', 0).text(label)
        .transition().delay(i * 80).duration(300).attr('opacity', 1);
      if (val !== null) {
        svg.append('text').attr('x', IX).attr('y', IY(i) + 9)
          .attr('text-anchor', 'middle').attr('fill', C.muted)
          .attr('font-size', 10).attr('font-family', 'monospace').attr('opacity', 0)
          .text(val.toFixed(2))
          .transition().delay(200 + i * 80).duration(300).attr('opacity', 1);
      }
    });

    // ── Connections with weight labels ────────────────────────────────────
    INPUTS.forEach(({ label }, i) => {
      if (label === '⋮') return;
      const x1 = IX + 22, y1 = IY(i), x2 = NX - NR;
      const wi = WV[i], wl = WL[i];

      svg.append('line')
        .attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', NY)
        .attr('stroke', C.conn).attr('stroke-width', 1.5)
        .attr('opacity', 0)
        .transition().delay(300 + i * 60).duration(400).attr('opacity', 1);

      if (wl) {
        const midX = (x1 + x2) / 2, midY = (y1 + NY) / 2;
        const wStr = (wi >= 0 ? '+' : '') + wi.toFixed(2);
        svg.append('rect')
          .attr('x', midX - 26).attr('y', midY - 11)
          .attr('width', 52).attr('height', 19)
          .attr('fill', 'rgba(157,78,221,0.15)').attr('stroke', 'rgba(157,78,221,0.5)')
          .attr('stroke-width', 1).attr('rx', 4).attr('opacity', 0)
          .transition().delay(500 + i * 60).duration(300).attr('opacity', 1);
        svg.append('text').attr('x', midX).attr('y', midY + 4)
          .attr('text-anchor', 'middle').attr('fill', C.secondary)
          .attr('font-size', 11).attr('font-family', 'monospace').attr('opacity', 0)
          .text(wl + '=' + wStr)
          .transition().delay(500 + i * 60).duration(300).attr('opacity', 1);
      }
    });

    // ── Bias node ──────────────────────────────────────────────────────────
    svg.append('circle').attr('cx', BIAS_X).attr('cy', BIAS_Y).attr('r', 20)
      .attr('fill', 'rgba(255,0,110,0.12)').attr('stroke', C.accent).attr('stroke-width', 1.5)
      .attr('opacity', 0).transition().delay(600).duration(300).attr('opacity', 1);
    svg.append('text').attr('x', BIAS_X).attr('y', BIAS_Y - 2)
      .attr('text-anchor', 'middle').attr('fill', C.accent).attr('font-size', 12)
      .attr('opacity', 0).text('b')
      .transition().delay(600).duration(300).attr('opacity', 1);
    svg.append('text').attr('x', BIAS_X).attr('y', BIAS_Y + 12)
      .attr('text-anchor', 'middle').attr('fill', C.accent)
      .attr('font-size', 9).attr('font-family', 'monospace').attr('opacity', 0)
      .text('+' + BIAS.toFixed(2))
      .transition().delay(620).duration(300).attr('opacity', 1);
    svg.append('line')
      .attr('x1', BIAS_X).attr('y1', BIAS_Y - 20)
      .attr('x2', NX - NR * 0.5).attr('y2', NY + NR * 0.72)
      .attr('stroke', 'rgba(255,0,110,0.35)').attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3')
      .attr('opacity', 0).transition().delay(650).duration(400).attr('opacity', 1);

    // ── Main neuron ────────────────────────────────────────────────────────
    svg.append('circle').attr('cx', NX).attr('cy', NY).attr('r', NR)
      .attr('fill', 'rgba(0,217,255,0.12)').attr('stroke', C.primary).attr('stroke-width', 2)
      .attr('class', 'perc-neuron')
      .attr('opacity', 0).transition().delay(700).duration(400).attr('opacity', 1);
    svg.append('text').attr('x', NX).attr('y', NY - 7)
      .attr('text-anchor', 'middle').attr('fill', C.primary).attr('font-size', 14)
      .attr('opacity', 0).text('Σ')
      .transition().delay(800).duration(300).attr('opacity', 1);
    svg.append('text').attr('x', NX).attr('y', NY + 11)
      .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 11)
      .attr('opacity', 0).text('σ(·)')
      .transition().delay(800).duration(300).attr('opacity', 1);

    // ── Output node ────────────────────────────────────────────────────────
    const outG = svg.append('g').attr('opacity', 0);
    outG.append('line')
      .attr('x1', NX + NR).attr('y1', NY).attr('x2', OX + 10).attr('y2', NY)
      .attr('stroke', C.primary).attr('stroke-width', 2).attr('marker-end', 'url(#arr3)');
    outG.append('circle').attr('cx', OX + 42).attr('cy', NY).attr('r', 26)
      .attr('fill', 'rgba(0,217,255,0.2)').attr('stroke', C.primary).attr('stroke-width', 2);
    outG.append('text').attr('x', OX + 42).attr('y', NY - 5)
      .attr('text-anchor', 'middle').attr('fill', C.primary).attr('font-size', 14).text('a');
    const aValTxt = outG.append('text').attr('x', OX + 42).attr('y', NY + 10)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 10).attr('font-family', 'monospace').text('');
    outG.transition().delay(900).duration(400).attr('opacity', 1);

    // ── Computation Panel ──────────────────────────────────────────────────
    const panelG = svg.append('g');

    // Background box
    panelG.append('rect')
      .attr('x', PX).attr('y', 10).attr('width', W - PX - 15).attr('height', H - 20)
      .attr('fill', 'rgba(0,217,255,0.04)').attr('stroke', 'rgba(0,217,255,0.25)')
      .attr('stroke-width', 1).attr('rx', 8)
      .attr('opacity', 0).transition().delay(950).duration(400).attr('opacity', 1);

    // Panel title
    panelG.append('text').attr('x', PX + (W - PX - 15) / 2).attr('y', 34)
      .attr('text-anchor', 'middle').attr('fill', C.primary)
      .attr('font-size', 12).attr('font-weight', 'bold').attr('font-family', 'monospace')
      .attr('opacity', 0).text('z  =  Σ wᵢ · xᵢ  +  b')
      .transition().delay(980).duration(400).attr('opacity', 1);

    // Header separator
    panelG.append('line')
      .attr('x1', PX + 10).attr('y1', 44).attr('x2', W - 22).attr('y2', 44)
      .attr('stroke', 'rgba(0,217,255,0.3)').attr('stroke-width', 1)
      .attr('opacity', 0).transition().delay(980).duration(300).attr('opacity', 1);

    // Vertical column divider (product | accumulator)
    panelG.append('line')
      .attr('x1', Csep).attr('y1', 44).attr('x2', Csep).attr('y2', H - 18)
      .attr('stroke', 'rgba(0,217,255,0.18)').attr('stroke-width', 0.8)
      .attr('opacity', 0).transition().delay(1000).duration(300).attr('opacity', 1);

    // Small column headers
    const hdrY = 57;
    [
      { x: CwEnd,  t: 'weight',    a: 'end'    },
      { x: Cmul,   t: '×',         a: 'middle' },
      { x: CxEnd,  t: 'input',     a: 'end'    },
      { x: Ceq,    t: '=',         a: 'middle' },
      { x: CpEnd,  t: 'wᵢ·xᵢ',    a: 'end'    },
      { x: CaccV,  t: 'Σ (running)', a: 'end'  },
    ].forEach(({ x, t, a }) => {
      panelG.append('text').attr('x', x).attr('y', hdrY)
        .attr('text-anchor', a).attr('fill', C.muted)
        .attr('font-size', 9).attr('opacity', 0).text(t)
        .transition().delay(1010).duration(300).attr('opacity', 0.65);
    });
    panelG.append('line')
      .attr('x1', PX + 10).attr('y1', 63).attr('x2', W - 22).attr('y2', 63)
      .attr('stroke', 'rgba(0,217,255,0.15)').attr('stroke-width', 0.6)
      .attr('opacity', 0).transition().delay(1010).duration(300).attr('opacity', 1);

    // ── Per-input rows (revealed on pulse arrival) ──────────────────────────
    const RY0 = 76, RYH = 40;
    const rowGs = [];
    let accum = 0;

    active.forEach((d, ri) => {
      const rowY   = RY0 + ri * RYH;
      accum        = Math.round((accum + d.prod) * 10000) / 10000;
      const prodStr = (d.prod >= 0 ? '+' : '') + d.prod.toFixed(3);
      const accumStr = (accum >= 0 ? '+' : '') + accum.toFixed(3);
      const wiStr   = (d.wi >= 0 ? '+' : '') + d.wi.toFixed(2);

      const g = panelG.append('g').attr('opacity', 0);

      // Weight col
      g.append('text').attr('x', CwEnd).attr('y', rowY + 5)
        .attr('text-anchor', 'end').attr('fill', C.secondary)
        .attr('font-size', 12).attr('font-family', 'monospace')
        .text(d.wl + '=' + wiStr);

      // × operator
      g.append('text').attr('x', Cmul).attr('y', rowY + 5)
        .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 13)
        .text('×');

      // Input col
      g.append('text').attr('x', CxEnd).attr('y', rowY + 5)
        .attr('text-anchor', 'end').attr('fill', C.primary)
        .attr('font-size', 12).attr('font-family', 'monospace')
        .text(d.label + '=' + d.val.toFixed(2));

      // =
      g.append('text').attr('x', Ceq).attr('y', rowY + 5)
        .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 13)
        .text('=');

      // Product (green = positive, red = negative)
      g.append('text').attr('x', CpEnd).attr('y', rowY + 5)
        .attr('text-anchor', 'end')
        .attr('fill', d.prod >= 0 ? '#4ade80' : '#f87171')
        .attr('font-size', 12).attr('font-family', 'monospace').attr('font-weight', 'bold')
        .text(prodStr);

      // Running accumulator
      g.append('text').attr('x', CaccV).attr('y', rowY + 5)
        .attr('text-anchor', 'end').attr('fill', '#fbbf24')
        .attr('font-size', 12).attr('font-family', 'monospace').attr('font-weight', 'bold')
        .text(accumStr);

      rowGs.push(g);
    });

    // ── Summary rows ────────────────────────────────────────────────────────
    const sumY   = RY0 + active.length * RYH;         // top of summary block
    const biasY  = sumY  + 34;
    const sep2Y  = biasY + 24;
    const zY     = sep2Y + 28;
    const aY     = zY   + 42;

    // Separator 1
    const sepLine1 = panelG.append('line')
      .attr('x1', PX + 10).attr('y1', sumY - 5)
      .attr('x2', W - 22).attr('y2', sumY - 5)
      .attr('stroke', 'rgba(0,217,255,0.4)').attr('stroke-width', 1).attr('opacity', 0);

    // Σwᵢxᵢ row
    const wsumG = panelG.append('g').attr('opacity', 0);
    wsumG.append('text').attr('x', CwEnd).attr('y', sumY + 20)
      .attr('text-anchor', 'end').attr('fill', C.muted)
      .attr('font-size', 11).attr('font-family', 'monospace').text('Σ wᵢxᵢ');
    wsumG.append('text').attr('x', Ceq).attr('y', sumY + 20)
      .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 13).text('=');
    wsumG.append('text').attr('x', CpEnd).attr('y', sumY + 20)
      .attr('text-anchor', 'end').attr('fill', '#fbbf24')
      .attr('font-size', 13).attr('font-family', 'monospace').attr('font-weight', 'bold')
      .text((wsum >= 0 ? '+' : '') + wsum.toFixed(3));

    // + b row
    const biasRowG = panelG.append('g').attr('opacity', 0);
    biasRowG.append('text').attr('x', CwEnd).attr('y', biasY + 5)
      .attr('text-anchor', 'end').attr('fill', C.accent)
      .attr('font-size', 12).attr('font-family', 'monospace').text('  + b');
    biasRowG.append('text').attr('x', Ceq).attr('y', biasY + 5)
      .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 13).text('=');
    biasRowG.append('text').attr('x', CpEnd).attr('y', biasY + 5)
      .attr('text-anchor', 'end').attr('fill', C.accent)
      .attr('font-size', 12).attr('font-family', 'monospace')
      .text('+' + BIAS.toFixed(2));

    // Separator 2
    const sepLine2 = panelG.append('line')
      .attr('x1', PX + 10).attr('y1', sep2Y)
      .attr('x2', W - 22).attr('y2', sep2Y)
      .attr('stroke', 'rgba(0,217,255,0.55)').attr('stroke-width', 1.5).attr('opacity', 0);

    // z = ... row
    const zG = panelG.append('g').attr('opacity', 0);
    zG.append('text').attr('x', CwEnd).attr('y', zY + 5)
      .attr('text-anchor', 'end').attr('fill', C.primary)
      .attr('font-size', 15).attr('font-weight', 'bold').text('z');
    zG.append('text').attr('x', Ceq).attr('y', zY + 5)
      .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 15).text('=');
    zG.append('text').attr('x', CpEnd).attr('y', zY + 5)
      .attr('text-anchor', 'end').attr('fill', C.primary)
      .attr('font-size', 15).attr('font-family', 'monospace').attr('font-weight', 'bold')
      .text((z >= 0 ? '+' : '') + z.toFixed(3));

    // a = σ(z) row
    const aG = panelG.append('g').attr('opacity', 0);
    aG.append('text').attr('x', CwEnd).attr('y', aY + 5)
      .attr('text-anchor', 'end').attr('fill', C.accent)
      .attr('font-size', 14).text('a = σ(z)');
    aG.append('text').attr('x', Ceq).attr('y', aY + 5)
      .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 15).text('=');
    aG.append('text').attr('x', CpEnd).attr('y', aY + 5)
      .attr('text-anchor', 'end').attr('fill', C.accent)
      .attr('font-size', 17).attr('font-family', 'monospace').attr('font-weight', 'bold')
      .text(aVal.toFixed(3));

    // σ type note
    const sigmaNoteG = panelG.append('g').attr('opacity', 0);
    sigmaNoteG.append('text').attr('x', PX + (W - PX - 15) / 2).attr('y', aY + 28)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 9).text('σ: sigmoid, ReLU, tanh, …');

    // ── Animated signal pulses ─────────────────────────────────────────────
    const PULSE_BASE = 1300;
    const PULSE_GAP  = 480;
    const PULSE_DUR  = 620;

    active.forEach((d, ri) => {
      const startT = PULSE_BASE + ri * PULSE_GAP;
      const x1 = IX + 22, y1 = IY(d.iy);

      setTimeout(() => {
        const pulse = svg.append('circle')
          .attr('cx', x1).attr('cy', y1).attr('r', 5)
          .attr('fill', C.primary).attr('opacity', 0.9);
        pulse.transition().duration(PULSE_DUR)
          .attr('cx', NX - NR).attr('cy', NY)
          .on('end', () => {
            pulse.transition().duration(200).attr('r', 2).attr('opacity', 0).remove();
            // Flash neuron
            svg.select('.perc-neuron')
              .transition().duration(140).attr('fill', 'rgba(0,217,255,0.45)')
              .transition().duration(360).attr('fill', 'rgba(0,217,255,0.12)');
            // Reveal corresponding computation row
            rowGs[ri].transition().duration(320).attr('opacity', 1);
          });
      }, startT);
    });

    // After all pulses: reveal summary rows in sequence
    const allDone = PULSE_BASE + (active.length - 1) * PULSE_GAP + PULSE_DUR + 250;

    setTimeout(() => {
      sepLine1.transition().duration(280).attr('opacity', 1);
      wsumG.transition().delay(180).duration(380).attr('opacity', 1);
    }, allDone);
    setTimeout(() => {
      biasRowG.transition().duration(380).attr('opacity', 1);
    }, allDone + 580);
    setTimeout(() => {
      sepLine2.transition().duration(260).attr('opacity', 1);
    }, allDone + 920);
    setTimeout(() => {
      zG.transition().duration(400).attr('opacity', 1);
      // Pulse from neuron to output
      const outPulse = svg.append('circle')
        .attr('cx', NX + NR).attr('cy', NY).attr('r', 5)
        .attr('fill', C.primary).attr('opacity', 0.9);
      outPulse.transition().duration(400)
        .attr('cx', OX + 10).attr('cy', NY)
        .on('end', () => outPulse.transition().duration(200).attr('opacity', 0).remove());
    }, allDone + 1150);
    setTimeout(() => {
      aG.transition().duration(400).attr('opacity', 1);
      sigmaNoteG.transition().delay(300).duration(400).attr('opacity', 1);
      aValTxt.text(aVal.toFixed(3));
    }, allDone + 1600);
  }

  // ── SLIDE: Activation Functions (ReLU & Sigmoid + bias effect) ───────────
  function initActivation(el) {
    // Remove any stale click handler from a previous init on this element
    if (el._clickHandler) {
      el.removeEventListener('click', el._clickHandler);
      el._clickHandler = null;
    }

    el.innerHTML = '';
    const W = 1100, H = 490;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%')
      .style('cursor', 'pointer');

    const defs = svg.append('defs');

    // Layout constants shared by both panels
    const PW  = 535, PH = H - 20;   // panel outer size
    const ML  = 52,  MR = 18;        // left / right margin inside panel
    const MT  = 66,  MB = 52;        // top / bottom margin inside panel
    const PW2 = PW - ML - MR;        // plot width  = 465
    const PH2 = PH - MT - MB;        // plot height = 352

    // Clip paths — rect coordinates are relative to each panel's <g> origin
    ['relu-clip', 'sigm-clip'].forEach(id => {
      defs.append('clipPath').attr('id', id)
        .append('rect')
          .attr('x', ML).attr('y', MT)
          .attr('width', PW2).attr('height', PH2);
    });

    // Build a compact SVG polyline path from an array of x values
    function polyPath(xs, xFn, yFn) {
      let d = '';
      for (let i = 0; i < xs.length; i++) {
        const px = xFn(xs[i]).toFixed(1);
        const py = yFn(xs[i]).toFixed(1);
        d += (i === 0 ? 'M' : 'L') + px + ',' + py;
      }
      return d;
    }

    function makePanel(panelX, type) {
      const g    = svg.append('g').attr('transform', `translate(${panelX},10)`);
      const isR  = (type === 'relu');
      const col  = isR ? C.primary : C.secondary;
      const clip = isR ? 'relu-clip' : 'sigm-clip';
      const fn   = isR
        ? z => Math.max(0, z)
        : z => 1 / (1 + Math.exp(-z));

      const xMin = -5, xMax = 5;
      const yMin = isR ? -0.6  : -0.08;
      const yMax = isR ?  5.6  :  1.12;

      const sx = v => ML + (v - xMin) / (xMax - xMin) * PW2;
      const sy = v => MT + PH2 - (v - yMin) / (yMax - yMin) * PH2;
      const clampY = v => Math.min(yMax, Math.max(yMin, v));

      // ── Static chrome ────────────────────────────────────────────────────
      g.append('rect')
        .attr('x', 0).attr('y', 0).attr('width', PW).attr('height', PH)
        .attr('fill', 'rgba(0,8,24,0.28)').attr('stroke', col + '44')
        .attr('stroke-width', 1).attr('rx', 8);

      g.append('text').attr('x', PW / 2).attr('y', 22)
        .attr('text-anchor', 'middle').attr('fill', col)
        .attr('font-size', 15).attr('font-weight', 'bold')
        .text(isR ? 'ReLU — Rectified Linear Unit' : 'Sigmoid — Logistic Function');
      g.append('text').attr('x', PW / 2).attr('y', 39)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 12).attr('font-family', 'monospace')
        .text(isR ? 'ReLU(z) = max(0, z)' : 'σ(z) = 1 / (1 + e⁻ᶻ)');

      // Subtitle — captured so we can update it on click
      const subtitleEl = g.append('text').attr('x', PW / 2).attr('y', 56)
        .attr('text-anchor', 'middle').attr('fill', 'rgba(173,181,189,0.55)')
        .attr('font-size', 10).attr('font-family', 'monospace')
        .text('z = 1·x + b   (b = 0  —  click to animate)');

      // Plot box
      g.append('rect')
        .attr('x', ML).attr('y', MT).attr('width', PW2).attr('height', PH2)
        .attr('fill', 'rgba(0,0,0,0.22)')
        .attr('stroke', 'rgba(255,255,255,0.1)').attr('stroke-width', 0.8);

      // y grid + tick labels
      const yTicks = isR ? [0,1,2,3,4,5] : [0,0.25,0.5,0.75,1.0];
      yTicks.forEach(v => {
        const yy = sy(v);
        if (yy < MT || yy > MT + PH2) return;
        g.append('line')
          .attr('x1', ML).attr('y1', yy).attr('x2', ML + PW2).attr('y2', yy)
          .attr('stroke', v === 0 ? 'rgba(255,255,255,0.22)' : 'rgba(255,255,255,0.06)')
          .attr('stroke-width', v === 0 ? 1 : 0.5);
        g.append('text').attr('x', ML - 5).attr('y', yy + 3.5)
          .attr('text-anchor', 'end').attr('fill', C.muted).attr('font-size', 9)
          .text(v % 1 === 0 ? v : v.toFixed(2));
      });

      // x ticks + grid
      const xTicks = [-4,-3,-2,-1,0,1,2,3,4];
      xTicks.forEach(v => {
        const xx = sx(v);
        if (v !== 0) {
          g.append('line')
            .attr('x1', xx).attr('y1', MT).attr('x2', xx).attr('y2', MT + PH2)
            .attr('stroke', 'rgba(255,255,255,0.04)').attr('stroke-width', 0.5);
        }
        g.append('line')
          .attr('x1', xx).attr('y1', MT + PH2)
          .attr('x2', xx).attr('y2', MT + PH2 + 5)
          .attr('stroke', C.muted).attr('stroke-width', 0.8);
        g.append('text').attr('x', xx).attr('y', MT + PH2 + 16)
          .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 9)
          .text(v);
      });

      g.append('text').attr('x', ML + PW2 / 2).attr('y', MT + PH2 + 30)
        .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 9)
        .text('input  x');
      g.append('text')
        .attr('transform', `translate(12, ${MT + PH2 / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 9)
        .text('output');

      // Sigmoid σ=0.5 reference line
      if (!isR) {
        g.append('line')
          .attr('x1', ML).attr('y1', sy(0.5)).attr('x2', ML + PW2).attr('y2', sy(0.5))
          .attr('stroke', 'rgba(255,0,110,0.3)').attr('stroke-width', 1)
          .attr('stroke-dasharray', '4,4');
        g.append('text').attr('x', ML - 5).attr('y', sy(0.5) + 3.5)
          .attr('text-anchor', 'end').attr('fill', 'rgba(255,0,110,0.6)')
          .attr('font-size', 8).text('0.5');
      }

      // x=2 reference line (w·x=2 example)
      g.append('line')
        .attr('x1', sx(2)).attr('y1', MT)
        .attr('x2', sx(2)).attr('y2', MT + PH2)
        .attr('stroke', 'rgba(251,191,36,0.18)').attr('stroke-width', 0.8)
        .attr('stroke-dasharray', '2,5');
      g.append('text').attr('x', sx(2) + 3).attr('y', MT - 4)
        .attr('fill', 'rgba(251,191,36,0.55)').attr('font-size', 8)
        .attr('font-family', 'monospace').text('x=2');

      // Bottom interpretation note
      g.append('text').attr('x', PW / 2).attr('y', MT + PH2 + 46)
        .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 9)
        .text(isR
          ? 'Bias shifts the threshold — where the neuron "turns on"'
          : 'Bias shifts the S-curve center — controls the neuron\'s baseline tendency');

      // ── Dynamic elements ──────────────────────────────────────────────────
      const N  = 350;
      const xs = Array.from({length: N + 1}, (_, i) => xMin + i * (xMax - xMin) / N);

      // Active (non-zero) curve
      const curvePath = g.append('path')
        .attr('fill', 'none').attr('stroke', col)
        .attr('stroke-width', 2.8).attr('clip-path', `url(#${clip})`);

      // ReLU dead-zone: visible solid flat line at y=0 where output = 0
      const deadPath = isR
        ? g.append('path').attr('fill', 'none')
            .attr('stroke', C.primary).attr('stroke-width', 2.8)
            .attr('stroke-dasharray', '8,5').attr('opacity', 0.55)
            .attr('clip-path', `url(#${clip})`)
        : null;

      // Threshold / midpoint vertical line
      const threshLine = g.append('line')
        .attr('stroke', C.accent).attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '6,3')
        .attr('y1', MT).attr('y2', MT + PH2);

      // Key point dot
      const keyDot = g.append('circle')
        .attr('r', 5.5).attr('fill', C.accent)
        .attr('stroke', '#fff').attr('stroke-width', 1.5);

      // Bias readout
      const biasLabel = g.append('text')
        .attr('text-anchor', 'end').attr('fill', C.accent)
        .attr('font-size', 14).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .attr('x', ML + PW2 - 8).attr('y', MT + 20);

      // Threshold/midpoint annotation
      const keyLabel = g.append('text')
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 10).attr('font-family', 'monospace').attr('font-weight', 'bold');
      const keyNote = g.append('text')
        .attr('text-anchor', 'middle').attr('fill', 'rgba(255,0,110,0.7)')
        .attr('font-size', 8.5).attr('font-family', 'monospace');

      // Example point at x=2
      const exDot = g.append('circle')
        .attr('r', 5).attr('fill', '#fbbf24')
        .attr('stroke', '#fff').attr('stroke-width', 1.2)
        .attr('clip-path', `url(#${clip})`);
      const exVal = g.append('text')
        .attr('fill', '#fbbf24').attr('font-size', 10).attr('font-family', 'monospace')
        .attr('font-weight', 'bold');

      // ── Update function ────────────────────────────────────────────────────
      function updateBias(b) {
        if (isR) {
          const thresh = -b;
          const onXs  = xs.filter(xv => xv >= thresh);
          const offXs = xs.filter(xv => xv <= thresh);
          curvePath.attr('d',
            onXs.length > 1
              ? polyPath(onXs, sx, xv => sy(Math.min(yMax, xv + b)))
              : null);
          deadPath.attr('d',
            offXs.length > 1
              ? polyPath(offXs, sx, () => sy(0))
              : null);
        } else {
          curvePath.attr('d',
            polyPath(xs, sx, xv => sy(clampY(fn(xv + b)))));
        }

        const kx    = -b;
        const kxScr = sx(kx);
        const vis   = kx >= xMin && kx <= xMax;
        const keyYScr = isR ? sy(0) : sy(0.5);

        threshLine.attr('x1', kxScr).attr('x2', kxScr)
          .attr('opacity', vis ? 0.9 : 0);
        keyDot.attr('cx', kxScr).attr('cy', keyYScr)
          .attr('opacity', vis ? 1 : 0);

        const labelAboveY = keyYScr - 14;
        keyLabel
          .attr('x', kxScr).attr('y', Math.max(MT + 12, labelAboveY))
          .text(isR ? `x = ${kx.toFixed(1)}` : `x = ${kx.toFixed(1)},  σ = 0.5`)
          .attr('opacity', vis ? 1 : 0);
        keyNote
          .attr('x', kxScr).attr('y', Math.max(MT + 24, labelAboveY + 13))
          .text(isR ? '← ReLU turns on here' : '← S-curve midpoint')
          .attr('opacity', vis ? 0.85 : 0);

        biasLabel.text('b = ' + (b >= 0 ? '+' : '') + b.toFixed(2));

        const exOut      = fn(2 + b);
        const exClamped  = clampY(exOut);
        const exOnScreen = exOut >= yMin && exOut <= yMax;
        const exCy       = sy(exClamped);
        exDot.attr('cx', sx(2)).attr('cy', exCy)
          .attr('opacity', exOnScreen ? 1 : 0.2);
        const exLabelY = Math.max(MT + 14, Math.min(MT + PH2 - 6, exCy - 8));
        exVal.attr('x', sx(2) + 9).attr('y', exLabelY)
          .text(exOut.toFixed(2))
          .attr('opacity', exOnScreen ? 1 : 0);
      }

      return { update: updateBias, subtitle: subtitleEl };
    }

    const relu = makePanel(10,  'relu');
    const sigm = makePanel(555, 'sigmoid');

    // ── Static initial render at b = 0 ────────────────────────────────────
    relu.update(0);
    sigm.update(0);

    // ── "Click to animate" hint ────────────────────────────────────────────
    const hintG = svg.append('g').attr('pointer-events', 'none');
    hintG.append('rect')
      .attr('x', W / 2 - 102).attr('y', H - 38)
      .attr('width', 204).attr('height', 26)
      .attr('rx', 6).attr('fill', 'rgba(0,0,0,0.55)')
      .attr('stroke', 'rgba(255,255,255,0.18)').attr('stroke-width', 0.8);
    hintG.append('text')
      .attr('x', W / 2).attr('y', H - 20)
      .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.65)')
      .attr('font-size', 12)
      .text('▶  Click anywhere to animate');

    // ── Animation loop (started on first click) ────────────────────────────
    let rafId = null, startTime = null;
    const PERIOD = 16000;   // 16 s for a full ±3 sine cycle (half original speed)

    function loop(ts) {
      if (!startTime) startTime = ts;
      const phase = ((ts - startTime) % PERIOD) / PERIOD;
      const b     = 3 * Math.sin(2 * Math.PI * phase);
      relu.update(b);
      sigm.update(b);
      rafId = requestAnimationFrame(loop);
    }

    el._clickHandler = function onClickStart() {
      if (rafId !== null) return;              // already animating
      hintG.transition().duration(350).attr('opacity', 0).remove();
      relu.subtitle.text('z = 1·x + b   (bias b sweeping  −3 → 0 → +3)');
      sigm.subtitle.text('z = 1·x + b   (bias b sweeping  −3 → 0 → +3)');
      svg.style('cursor', 'default');
      rafId = requestAnimationFrame(loop);
    };
    el.addEventListener('click', el._clickHandler);

    el._cancelAnim = () => {
      if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
      startTime = null;
    };
  }

  // ── SLIDE 4: Full Network ────────────────────────────────────────────────
  function initFullNetwork(el) {
    loadSample().then(() => _renderFullNetwork(el));
  }
  function _renderFullNetwork(el) {
    el.innerHTML = '';
    const W = 1100, H = 520;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%');

    addArrowDef(svg, 'arr4', C.primary);

    const LAYERS = [
      { label: 'Input',    sub: '784',  n: 22, color: C.primary,   x: 80,  r: 6, special: true },
      { label: 'Hidden 1', sub: '64',   n: 64, color: C.secondary, x: 330, r: 3 },
      { label: 'Hidden 2', sub: '32',   n: 32, color: C.secondary, x: 580, r: 5 },
      { label: 'Output',   sub: '10',   n: 10, color: C.accent,    x: 830, r: 9 },
    ];

    const R = 9; // default / output radius used for pulse animation
    // Hot colormap identical to Training slide: black → red → orange → yellow → white
    const hotColor = t => {
      const r  = Math.min(1, t * 3);
      const g  = Math.min(1, Math.max(0, t * 3 - 1));
      const bv = Math.min(1, Math.max(0, t * 3 - 2));
      return `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(bv*255)})`;
    };
    const nodeY = (layerN, idx) => {
      const span = H - 80;
      const spacing = span / (layerN - 1);
      return 40 + idx * spacing;
    };

    // Use the shared input samples computed at module load (same pixels as MNIST→Vector)
    const inputPixels = INPUT_SAMPLES;

    // Draw connections first (behind nodes)
    for (let li = 0; li < LAYERS.length - 1; li++) {
      const L1 = LAYERS[li], L2 = LAYERS[li + 1];
      for (let a = 0; a < L1.n; a++) {
        for (let b = 0; b < L2.n; b++) {
          svg.append('line')
            .attr('x1', L1.x).attr('y1', nodeY(L1.n, a))
            .attr('x2', L2.x).attr('y2', nodeY(L2.n, b))
            .attr('stroke', C.conn).attr('stroke-width', 0.4)
            .attr('class', `conn-${li}-${a} conn-to-${li+1}-${b}`)
            .attr('opacity', 0)
            .transition().delay(li * 120 + a * 8).duration(300).attr('opacity', 1);
        }
      }
    }

    // Draw nodes
    LAYERS.forEach((layer, li) => {
      // Layer label
      svg.append('text').attr('x', layer.x).attr('y', H - 8)
        .attr('text-anchor', 'middle').attr('fill', layer.color)
        .attr('font-size', 12).attr('font-weight', 'bold')
        .attr('opacity', 0).text(layer.label)
        .transition().delay(li * 200).duration(300).attr('opacity', 1);
      svg.append('text').attr('x', layer.x).attr('y', H - 8 + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 10).attr('opacity', 0).text('n=' + layer.sub)
        .transition().delay(li * 200 + 50).duration(300).attr('opacity', 1);

      for (let i = 0; i < layer.n; i++) {
        const cy = nodeY(layer.n, i);
        const isOutput = li === LAYERS.length - 1;

        const lr = layer.r;

        // Glow halo (skip for dense hidden layers where nodes are tightly packed)
        if (lr >= 5) {
          svg.append('circle')
            .attr('cx', layer.x).attr('cy', cy).attr('r', lr * 2.2)
            .attr('fill', 'rgba(255,80,0,0)').attr('stroke', 'none')
            .attr('class', `halo-${li}-${i}`)
            .attr('opacity', 0);
        }

        svg.append('circle')
          .attr('cx', layer.x).attr('cy', cy).attr('r', lr)
          .attr('fill', `rgba(${li===0?'0,217,255':li===3?'255,0,110':'157,78,221'},0.15)`)
          .attr('stroke', layer.color).attr('stroke-width', isOutput ? 1.5 : 0.8)
          .attr('class', `node-${li}-${i}`)
          .attr('opacity', 0)
          .transition().delay(li * 200 + i * 5).duration(200).attr('opacity', 1);

        // Output layer digit labels
        if (isOutput) {
          svg.append('text')
            .attr('x', layer.x + lr + 6).attr('y', cy + 4)
            .attr('fill', C.accent).attr('font-size', 9)
            .attr('opacity', 0).text(i)
            .transition().delay(li * 200 + i * 5 + 80).duration(200).attr('opacity', 1);
        }

        // Input layer: pixel value labels (left of node, revealed by forward pass)
        if (li === 0) {
          const val = inputPixels[i] ? inputPixels[i].v : 0;
          svg.append('text')
            .attr('x', layer.x - lr - 7).attr('y', cy + 4)
            .attr('text-anchor', 'end')
            .attr('fill', C.primary).attr('font-size', 9).attr('font-family', 'monospace')
            .attr('class', `inp-val-${i}`)
            .attr('opacity', 0)
            .text(val.toFixed(2));
        }
      }

      // "..." indicator for input layer (show only subset of 784)
      if (layer.special) {
        svg.append('text').attr('x', layer.x).attr('y', nodeY(layer.n, layer.n - 1) + 20)
          .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 11)
          .attr('opacity', 0).text('(22 of 784 shown)')
          .transition().delay(100).duration(300).attr('opacity', 0.7);
      }
    });

    // Animated forward pass — wave of highlighted nodes + connections
    let baseDelay = LAYERS.length * 200 + 400;

    // ── Activations and weights from the trained network ─────────────────────
    // snap.W1 is a 28×64 slice (center column of the image); W2/W3 are full.
    const NH1 = 64, NH2 = 32, NO = 10;
    const snap = window.TRAINING_SNAPSHOTS && window.TRAINING_SNAPSHOTS.length
      ? window.TRAINING_SNAPSHOTS[window.TRAINING_SNAPSHOTS.length - 1]
      : null;

    // Hidden activations come directly from the sample (exported by train_mlp.py)
    const h1Full = (_sample.a1 && _sample.a1.length === NH1) ? _sample.a1 : new Array(NH1).fill(0);
    const h2Full = (_sample.a2 && _sample.a2.length === NH2) ? _sample.a2 : new Array(NH2).fill(0);

    // Use all neurons in each hidden layer (matches LAYERS n=64 and n=32)
    const h1Idx = Array.from({length: NH1}, (_, i) => i);
    const h2Idx = Array.from({length: NH2}, (_, i) => i);

    // Normalise hidden activations to [0, 1] for hotColor mapping
    const maxH1 = h1Full.reduce((m, v) => Math.max(m, v), 1e-9);
    const maxH2 = h2Full.reduce((m, v) => Math.max(m, v), 1e-9);

    // Global weight max for normalisation
    let globalMaxW = 1e-9;
    if (snap) {
      for (const w of snap.W1) { const a = Math.abs(w); if (a > globalMaxW) globalMaxW = a; }
      for (const w of snap.W2) { const a = Math.abs(w); if (a > globalMaxW) globalMaxW = a; }
      for (const w of snap.W3) { const a = Math.abs(w); if (a > globalMaxW) globalMaxW = a; }
    }

    // Weights for displayed connections (cyan = positive, purple = negative).
    // snap.W1 is stored as 28 rows of the center image column (col 14), indexed
    // as W1[row * NH1 + hiddenNeuron].  Map each displayed input pixel to its
    // image row and use that row's center-column weight as a proxy.
    const wMag = {}, wPos = {};
    if (snap) {
      for (let a = 0; a < inputPixels.length; a++) {
        const row = Math.floor(inputPixels[a].idx / 28);   // image row 0..27
        for (let b = 0; b < h1Idx.length; b++) {
          const w = snap.W1[row * NH1 + h1Idx[b]];
          wMag[`0-${a}-${b}`] = Math.min(1, Math.pow(Math.abs(w) / globalMaxW, 0.6));
          wPos[`0-${a}-${b}`] = w >= 0;
        }
      }
      for (let a = 0; a < h1Idx.length; a++)
        for (let b = 0; b < h2Idx.length; b++) {
          const w = snap.W2[h1Idx[a] * NH2 + h2Idx[b]];
          wMag[`1-${a}-${b}`] = Math.min(1, Math.pow(Math.abs(w) / globalMaxW, 0.6));
          wPos[`1-${a}-${b}`] = w >= 0;
        }
      for (let a = 0; a < h2Idx.length; a++)
        for (let b = 0; b < NO; b++) {
          const w = snap.W3[h2Idx[a] * NO + b];
          wMag[`2-${a}-${b}`] = Math.min(1, Math.pow(Math.abs(w) / globalMaxW, 0.6));
          wPos[`2-${a}-${b}`] = w >= 0;
        }
    }

    // Per-layer activations for the selected MNIST digit
    const pred  = _sample.predicted ?? MNIST_LABEL ?? 0;
    const acts0 = inputPixels.map(p => p.v);
    const acts1 = h1Full.map(v => v / maxH1);
    const acts2 = h2Full.map(v => v / maxH2);
    const acts3 = _sample.probs || Array.from({length: 10}, (_, i) => i === pred ? 0.85 : 0.015);
    const allActs = [acts0, acts1, acts2, acts3];

    function animateForwardPass() {
      LAYERS.forEach((layer, li) => {
        const delay = baseDelay + li * 400;
        // Stagger proportional to layer size so total stagger ≈ 200ms regardless of n
        const nodeStep = Math.max(1, Math.floor(200 / layer.n));

        for (let i = 0; i < layer.n; i++) {
          const act = Math.min(1, allActs[li][i] ?? 0);
          const nd  = delay + i * nodeStep;

          // All layers: hotColor ∝ activation/pixel value
          svg.select(`.node-${li}-${i}`)
            .transition().delay(nd).duration(350)
            .attr('fill', hotColor(Math.max(0.08, act)));

          // Glow halo for active neurons (only where halos were drawn: r >= 5)
          if (act > 0.05 && layer.r >= 5) {
            const glowRgb   = li === LAYERS.length - 1 ? '255,140,0' : '255,80,0';
            const glowAlpha = (act * 0.20).toFixed(2);
            svg.select(`.halo-${li}-${i}`)
              .transition().delay(nd).duration(600)
              .attr('r', layer.r * (1.8 + act * 1.4))
              .attr('fill', `rgba(${glowRgb},${glowAlpha})`)
              .attr('opacity', 1);
          }

          // Input layer: reveal pixel value label
          if (li === 0) {
            svg.select(`.inp-val-${i}`)
              .transition().delay(nd).duration(300)
              .attr('opacity', 1);
          }
        }

        // Incoming connections arrive at weight-brightness; cyan = positive, purple = negative
        if (li > 0) {
          const prevLayer = LAYERS[li - 1];
          const connStep  = Math.max(1, Math.floor(200 / prevLayer.n));
          for (let a = 0; a < prevLayer.n; a++) {
            for (let b = 0; b < layer.n; b++) {
              const key   = `${li-1}-${a}-${b}`;
              const mag   = wMag[key] ?? 0;
              const alpha = (mag * 0.80).toFixed(2);
              const sw    = (0.1  + mag * 2.2).toFixed(2);
              const rgb   = wPos[key] ? '0,217,255' : '157,78,221';
              svg.selectAll(`.conn-${li-1}-${a}.conn-to-${li}-${b}`)
                .transition().delay(delay + a * connStep).duration(450)
                .attr('stroke', `rgba(${rgb},${alpha})`)
                .attr('stroke-width', sw);
            }
          }
        }
      });

      // Pulse predicted output node to draw attention, then hold
      setTimeout(() => {
        svg.select(`.node-3-${pred}`)
          .transition().duration(300).attr('r', R + 3)
          .transition().duration(600).attr('r', R);
      }, baseDelay + LAYERS.length * 350 + 200);
    }

    setTimeout(animateForwardPass, 50);

    // Legend
    const legG = svg.append('g').attr('opacity', 0);
    legG.append('rect').attr('x', 920).attr('y', 40).attr('width', 165).attr('height', 120)
      .attr('fill', 'rgba(0,0,0,0.3)').attr('stroke', 'rgba(255,255,255,0.1)').attr('rx', 8);
    const legend = [
      { color: C.primary,   label: 'Input  (784)' },
      { color: C.secondary, label: 'Hidden (2×)' },
      { color: C.accent,    label: 'Output (10)' },
    ];
    legend.forEach(({ color, label }, i) => {
      legG.append('circle').attr('cx', 936).attr('cy', 62 + i * 30).attr('r', 7)
        .attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.5);
      legG.append('text').attr('x', 952).attr('y', 67 + i * 30)
        .attr('fill', C.text).attr('font-size', 11).text(label);
    });
    legG.transition().delay(1200).duration(400).attr('opacity', 1);

    // Forward pass label
    svg.append('text').attr('x', W / 2).attr('y', 22)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 12).attr('opacity', 0)
      .text('Forward pass: activations propagate left → right')
      .transition().delay(baseDelay).duration(400).attr('opacity', 1);

    // ── Pixelated MNIST digit (lower-right, below legend box) ──────────────
    // Render the 28×28 pixel array to an offscreen canvas, then insert as
    // a scaled SVG <image> with image-rendering: pixelated for crisp blocks.
    const CELL    = 5;                          // px per pixel cell
    const DSIZE   = 28 * CELL;                  // 140 px rendered size
    // Centre inside the legend column (x=920, w=165)
    const DIGIT_X = 920 + Math.round((165 - DSIZE) / 2);  // ≈ 933
    const DIGIT_Y = 175;

    const offC = document.createElement('canvas');
    offC.width = offC.height = 28;
    const oct  = offC.getContext('2d');
    const imd  = oct.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = Math.round((PIXEL_DATA[i] || 0) * 255);
      imd.data[i * 4]     = v;
      imd.data[i * 4 + 1] = v;
      imd.data[i * 4 + 2] = v;
      imd.data[i * 4 + 3] = 255;
    }
    oct.putImageData(imd, 0, 0);

    const digitG = svg.append('g').attr('opacity', 0);

    // Section header
    digitG.append('text')
      .attr('x', DIGIT_X + DSIZE / 2).attr('y', DIGIT_Y - 8)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 10).text('Input digit');

    // Cyan border frame
    digitG.append('rect')
      .attr('x', DIGIT_X - 1).attr('y', DIGIT_Y - 1)
      .attr('width', DSIZE + 2).attr('height', DSIZE + 2)
      .attr('fill', 'none')
      .attr('stroke', 'rgba(0,217,255,0.45)').attr('stroke-width', 1)
      .attr('rx', 2);

    // The pixelated image itself
    digitG.append('image')
      .attr('href', offC.toDataURL())
      .attr('x', DIGIT_X).attr('y', DIGIT_Y)
      .attr('width', DSIZE).attr('height', DSIZE)
      .attr('style', 'image-rendering:pixelated;image-rendering:crisp-edges;');

    // True label and prediction below the image
    const correct = (pred === MNIST_LABEL);
    digitG.append('text')
      .attr('x', DIGIT_X + DSIZE / 2).attr('y', DIGIT_Y + DSIZE + 16)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 10)
      .text(`True label: ${MNIST_LABEL}`);
    digitG.append('text')
      .attr('x', DIGIT_X + DSIZE / 2).attr('y', DIGIT_Y + DSIZE + 32)
      .attr('text-anchor', 'middle')
      .attr('fill', correct ? '#4ade80' : '#f87171')
      .attr('font-size', 12).attr('font-weight', 'bold')
      .text(`Predicted: ${pred}  ${correct ? '✓' : '✗'}`);

    digitG.transition().delay(1200).duration(600).attr('opacity', 1);

  }

  // ── SLIDE: Training Network ─────────────────────────────────────────────
  function initTrainingNetwork(el) {
    loadSample().then(() => _renderTrainingNetwork(el));
  }

  function _renderTrainingNetwork(el) {
    el.innerHTML = '';
    const snaps = window.TRAINING_SNAPSHOTS;
    if (!snaps || !snaps.length) {
      el.innerHTML = '<p style="color:#adb5bd;padding:20px">No training snapshot data found.</p>';
      return;
    }

    // ── Dimensions ────────────────────────────────────────────────────────────
    const NI=28, NH1=64, NH2=32, NO=10;
    const CW = el.clientWidth  || 1100;
    const CH = el.clientHeight || 505;
    const DPR = window.devicePixelRatio || 1;

    const canvas = document.createElement('canvas');
    canvas.width  = CW * DPR;
    canvas.height = CH * DPR;
    canvas.style.width  = CW + 'px';
    canvas.style.height = CH + 'px';
    el.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    ctx.scale(DPR, DPR);

    // ── Layer layout ──────────────────────────────────────────────────────────
    // Leave right 240px for chart/readout
    const NET_W = CW - 250;
    const LAYERS = [
      { n: NI,  x: 55,                color: C.primary,   R: 5,   label: 'Input',    sub: '784 (28 shown)' },
      { n: NH1, x: NET_W * 0.35,      color: C.secondary, R: 3,   label: 'Hidden 1', sub: '64'  },
      { n: NH2, x: NET_W * 0.65,      color: C.secondary, R: 5,   label: 'Hidden 2', sub: '32'  },
      { n: NO,  x: NET_W * 0.95,      color: C.accent,    R: 7,   label: 'Output',   sub: '10'  },
    ];

    const NET_TOP = 16, NET_BOT = CH - 36;
    const nodeY = (n, i) => NET_TOP + i * (NET_BOT - NET_TOP) / (n - 1);

    // ── Weight / bias lookup ──────────────────────────────────────────────────
    const getW = (snap, li, a, b) => {
      if (li === 0) return snap.W1[a * NH1 + b];   // W1: (28,64)
      if (li === 1) return snap.W2[a * NH2 + b];   // W2: (64,32)
      if (li === 2) return snap.W3[a * NO  + b];   // W3: (32,10)
    };
    const getB = (snap, li, i) => {
      if (li === 1) return snap.b1[i];
      if (li === 2) return snap.b2[i];
      if (li === 3) return snap.b3[i];
      return 0;
    };

    // Normalise weights against the FINAL epoch's max — so the initial near-zero
    // state is dark and the trained state is bright (true visual growth).
    const finalSnap = snaps[snaps.length - 1];
    const globalMaxW = Math.max(
      ...finalSnap.W1.map(Math.abs),
      ...finalSnap.W2.map(Math.abs),
      ...finalSnap.W3.map(Math.abs)
    ) || 1;

    // Bias range across ALL epochs and layers → used for hot-colormap normalisation
    let minBias = Infinity, maxBias = -Infinity;
    snaps.forEach(s => {
      [...s.b1, ...s.b2, ...s.b3].forEach(b => {
        if (b < minBias) minBias = b;
        if (b > maxBias) maxBias = b;
      });
    });
    const biasRange = (maxBias - minBias) || 1;
    const biasNorm  = b => (b - minBias) / biasRange;  // → [0, 1]

    // Hot temperature colormap: black → red → orange → yellow → white
    const hotColor = t => {
      const r = Math.min(1, t * 3);
      const g = Math.min(1, Math.max(0, t * 3 - 1));
      const bv = Math.min(1, Math.max(0, t * 3 - 2));
      return `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(bv*255)})`;
    };

    // Weight → line style, normalised so epoch-0 lines are near-invisible
    const wStyle = (w) => {
      const norm = Math.abs(w) / globalMaxW;
      const t    = Math.pow(norm, 0.6);
      return {
        lw:    0.1 + t * 2.2,
        alpha: t * 0.8,
        rgb:   w >= 0 ? '0,217,255' : '157,78,221',
      };
    };

    // ── Chart geometry ────────────────────────────────────────────────────────
    const CX = NET_W + 28, CY = 14, CWIDTH = CW - CX - 10, CHEIGHT = 120;
    const losses = snaps.map(s => s.loss);
    const maxL = Math.max(...losses), minL = Math.min(...losses);
    const chartX = (i) => CX + 10 + i * (CWIDTH - 20) / (snaps.length - 1);
    const chartYL = (v) => CY + CHEIGHT - 8 - (v - minL) / (maxL - minL) * (CHEIGHT - 18);
    const chartYA = (v) => CY + CHEIGHT - 8 - (v / 100) * (CHEIGHT - 18);

    // ── Per-epoch phase fractions ─────────────────────────────────────────────
    // Each epoch transition (t ∈ [0,1]) is split into 3 visual phases:
    //   [0, FWD_END)  — forward pass: cyan wave sweeps left→right
    //   [FWD_END, BWD_END) — backward pass: orange wave sweeps right→left
    //   [BWD_END, 1]  — weight update: weights/biases lerp from s0 to s1
    const FWD_END = 0.33, BWD_END = 0.66;

    // ── Draw one frame ────────────────────────────────────────────────────────
    // si = snapshot index, t ∈ [0,1] = progress within this epoch transition
    function drawFrame(si, t) {
      ctx.clearRect(0, 0, CW, CH);

      const s0 = snaps[si];
      const s1 = snaps[Math.min(si + 1, snaps.length - 1)];

      // Determine sub-phase and weight lerp factor
      let phase, phaseT, wt;
      if (t < FWD_END) {
        phase  = 'forward';
        phaseT = t / FWD_END;
        wt     = 0;
      } else if (t < BWD_END) {
        phase  = 'backward';
        phaseT = (t - FWD_END) / (BWD_END - FWD_END);
        wt     = 0;
      } else {
        phase  = 'update';
        phaseT = (t - BWD_END) / (1 - BWD_END);
        wt     = phaseT;
      }

      const lerpW = (a, b) => a + (b - a) * wt;

      // Wave front x-position
      const minX = LAYERS[0].x, maxX = LAYERS[LAYERS.length - 1].x;
      const waveX  = phase === 'forward'  ? minX + phaseT * (maxX - minX)
                   : phase === 'backward' ? maxX - phaseT * (maxX - minX)
                   : null;
      const waveSpread = (maxX - minX) * 0.22;  // highlight half-width

      const layerGlow = (lx) =>
        waveX === null ? 0 : Math.max(0, 1 - Math.abs(lx - waveX) / waveSpread);

      // ── Connections ──
      for (let li = 0; li < LAYERS.length - 1; li++) {
        const L1 = LAYERS[li], L2 = LAYERS[li + 1];
        const midX = (L1.x + L2.x) / 2;
        const glow = layerGlow(midX);

        for (let a = 0; a < L1.n; a++) {
          const y1 = nodeY(L1.n, a);
          for (let b = 0; b < L2.n; b++) {
            const w  = lerpW(getW(s0, li, a, b), getW(s1, li, a, b));
            const st = wStyle(w);
            let rgb   = st.rgb;
            let alpha = st.alpha;
            let lw    = st.lw;
            if (glow > 0) {
              rgb   = phase === 'forward' ? '0,217,255' : '255,140,0';
              alpha = Math.min(1, st.alpha + glow * 0.25);
              lw    = st.lw + glow * 0.3;
            }
            ctx.beginPath();
            ctx.moveTo(L1.x, y1);
            ctx.lineTo(L2.x, nodeY(L2.n, b));
            ctx.strokeStyle = `rgba(${rgb},${alpha.toFixed(3)})`;
            ctx.lineWidth   = lw;
            ctx.stroke();
          }
        }
      }

      // ── Nodes ──
      LAYERS.forEach((layer, li) => {
        const glow = layerGlow(layer.x);
        for (let i = 0; i < layer.n; i++) {
          const cy = nodeY(layer.n, i);

          // Outer glow halo
          if (glow > 0.15) {
            ctx.beginPath();
            ctx.arc(layer.x, cy, layer.R * (1 + glow * 1.4), 0, Math.PI * 2);
            ctx.fillStyle = phase === 'forward'
              ? `rgba(0,217,255,${(glow * 0.12).toFixed(3)})`
              : `rgba(255,140,0,${(glow * 0.10).toFixed(3)})`;
            ctx.fill();
          }

          ctx.beginPath();
          ctx.arc(layer.x, cy, layer.R, 0, Math.PI * 2);
          if (li === 0) {
            ctx.fillStyle = glow > 0.15
              ? `rgba(0,217,255,${(0.15 + glow * 0.20).toFixed(3)})`
              : 'rgba(0,217,255,0.15)';
          } else {
            const bv = lerpW(getB(s0, li, i), getB(s1, li, i));
            ctx.fillStyle = hotColor(biasNorm(bv));
          }
          ctx.strokeStyle = layer.color;
          ctx.lineWidth   = 0.8;
          ctx.fill();
          ctx.stroke();
        }
      });

      // Output digit labels
      for (let i = 0; i < NO; i++) {
        ctx.fillStyle = C.accent;
        ctx.font      = '9px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(String(i), LAYERS[3].x + LAYERS[3].R + 4, nodeY(NO, i) + 3);
      }

      // Layer labels
      LAYERS.forEach((L) => {
        ctx.fillStyle = L.color;
        ctx.font      = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(L.label, L.x, CH - 20);
        ctx.fillStyle = C.muted;
        ctx.font      = '9px sans-serif';
        ctx.fillText('n=' + L.sub, L.x, CH - 9);
      });

      // ── Model info table (lower-right corner) ─────────────────────────────
      const T_HDR = 18, T_ROW = 15, T_PAD = 7;
      const TX = CX + 2, TW = CWIDTH - 4;
      const infoRows = [
        ['Layers',      '784 → 64 → 32 → 10'],
        ['Weights',     '50,176 + 2,048 + 320'],
        ['Biases',      '64 + 32 + 10 = 106'],
        ['Activation',  'ReLU / Softmax'],
        ['Weight init', 'He (W₁) · σ=0.05 (W₂,W₃)'],
        ['Backprop',    'SGD · lr=0.005 · bs=256'],
        ['Loss fn',     'Cross-entropy'],
      ];
      const TH = T_HDR + infoRows.length * T_ROW + T_PAD * 2;
      const TY = CH - TH - 10;

      ctx.fillStyle   = 'rgba(8,10,18,0.86)';
      ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      ctx.lineWidth   = 0.7;
      ctx.beginPath();
      roundRect(ctx, TX, TY, TW, TH, 5);
      ctx.fill(); ctx.stroke();

      // Header
      ctx.fillStyle = C.primary;
      ctx.font      = 'bold 9px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('Model Parameters', TX + T_PAD, TY + T_HDR - 4);

      // Divider line
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth   = 0.5;
      ctx.beginPath();
      ctx.moveTo(TX + 1, TY + T_HDR);
      ctx.lineTo(TX + TW - 1, TY + T_HDR);
      ctx.stroke();

      // Rows
      const COL2 = TX + T_PAD + 68;
      infoRows.forEach(([lbl, val], ri) => {
        const ry = TY + T_HDR + T_PAD + (ri + 0.8) * T_ROW;
        ctx.font      = '8px sans-serif';
        ctx.fillStyle = C.muted;
        ctx.textAlign = 'left';
        ctx.fillText(lbl, TX + T_PAD, ry);
        ctx.font      = '8px monospace';
        ctx.fillStyle = 'rgba(210,215,228,0.88)';
        ctx.fillText(val, COL2, ry);
      });

      // ── Chart background ──
      ctx.fillStyle   = 'rgba(0,0,0,0.35)';
      ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      ctx.lineWidth   = 0.8;
      ctx.beginPath();
      roundRect(ctx, CX, CY, CWIDTH, CHEIGHT, 5);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = C.muted;
      ctx.font      = '9px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Loss / Accuracy', CX + CWIDTH / 2, CY + 12);

      // Loss + accuracy curves up to current snapshot
      if (si >= 1) {
        ctx.beginPath();
        for (let i = 0; i <= si; i++) {
          const px = chartX(i), py = chartYL(snaps[i].loss);
          i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        }
        ctx.strokeStyle = C.accent;
        ctx.lineWidth   = 1.5;
        ctx.stroke();

        ctx.beginPath();
        for (let i = 0; i <= si; i++) {
          const px = chartX(i), py = chartYA(snaps[i].acc * 100);
          i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        }
        ctx.setLineDash([3, 2]);
        ctx.strokeStyle = C.primary;
        ctx.lineWidth   = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Chart legend
      ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(CX+10, CY+CHEIGHT-4); ctx.lineTo(CX+22, CY+CHEIGHT-4); ctx.stroke();
      ctx.fillStyle = C.muted; ctx.font = '8px sans-serif'; ctx.textAlign = 'left';
      ctx.fillText('loss', CX+25, CY+CHEIGHT-1);
      ctx.setLineDash([3,2]); ctx.strokeStyle = C.primary;
      ctx.beginPath(); ctx.moveTo(CX+54, CY+CHEIGHT-4); ctx.lineTo(CX+66, CY+CHEIGHT-4); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText('acc %', CX+69, CY+CHEIGHT-1);

      // ── Readout ──
      const epochNum  = s0.epoch;
      const lossNow   = s0.loss + (s1.loss - s0.loss) * wt;
      const accNow    = (s0.acc  + (s1.acc  - s0.acc)  * wt) * 100;
      const phaseLabel = phase === 'forward'  ? '→ Forward pass'
                       : phase === 'backward' ? '← Backward pass'
                       : '⟳ Weight update';
      const phaseColor = phase === 'forward'  ? C.primary
                       : phase === 'backward' ? '#ff8c00'
                       : C.secondary;

      ctx.textAlign = 'center';
      ctx.fillStyle = C.primary;
      ctx.font      = 'bold 13px sans-serif';
      ctx.fillText(`Epoch ${epochNum}`, CX + CWIDTH / 2, CY + CHEIGHT + 20);
      ctx.fillStyle = phaseColor;
      ctx.font      = 'bold 11px sans-serif';
      ctx.fillText(phaseLabel, CX + CWIDTH / 2, CY + CHEIGHT + 36);
      ctx.fillStyle = C.accent;
      ctx.font      = '11px monospace';
      ctx.fillText(`loss: ${lossNow.toFixed(4)}`, CX + CWIDTH / 2, CY + CHEIGHT + 52);
      ctx.fillStyle = C.primary;
      ctx.fillText(`acc:  ${accNow.toFixed(1)}%`,  CX + CWIDTH / 2, CY + CHEIGHT + 68);

      // Connection legend
      ctx.textAlign = 'left';
      ctx.fillStyle = C.muted;
      ctx.font      = '9px sans-serif';
      const legY = CY + CHEIGHT + 90;
      ctx.strokeStyle = 'rgba(0,217,255,0.8)'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(CX, legY); ctx.lineTo(CX+16, legY); ctx.stroke();
      ctx.fillText('positive', CX+19, legY+3);
      ctx.strokeStyle = 'rgba(157,78,221,0.8)';
      ctx.beginPath(); ctx.moveTo(CX+76, legY); ctx.lineTo(CX+92, legY); ctx.stroke();
      ctx.fillText('negative', CX+95, legY+3);

      // Bias hot-colormap gradient swatch
      const swY = legY + 18;
      const swW = CWIDTH - 20, swH = 9;
      const grad = ctx.createLinearGradient(CX+10, swY, CX+10+swW, swY);
      grad.addColorStop(0,    'rgb(0,0,0)');
      grad.addColorStop(0.33, 'rgb(255,0,0)');
      grad.addColorStop(0.67, 'rgb(255,200,0)');
      grad.addColorStop(1,    'rgb(255,255,255)');
      ctx.fillStyle = grad;
      ctx.fillRect(CX+10, swY, swW, swH);
      ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 0.5;
      ctx.strokeRect(CX+10, swY, swW, swH);
      ctx.fillStyle = C.muted; ctx.font = '8px sans-serif'; ctx.textAlign = 'left';
      ctx.fillText(`bias  ${minBias.toFixed(2)}`, CX+10, swY+swH+9);
      ctx.textAlign = 'right';
      ctx.fillText(maxBias.toFixed(2), CX+10+swW, swY+swH+9);
      ctx.textAlign = 'center';
      ctx.fillText('node bias (hot scale)', CX+10+swW/2, swY+swH+9);
    }

    // ── roundRect helper ──────────────────────────────────────────────────────
    function roundRect(ctx, x, y, w, h, r) {
      ctx.moveTo(x+r, y);
      ctx.lineTo(x+w-r, y); ctx.arcTo(x+w,y,  x+w,y+r,  r);
      ctx.lineTo(x+w, y+h-r); ctx.arcTo(x+w,y+h,x+w-r,y+h,r);
      ctx.lineTo(x+r, y+h); ctx.arcTo(x,y+h,  x,y+h-r,  r);
      ctx.lineTo(x, y+r);   ctx.arcTo(x,y,    x+r,y,    r);
      ctx.closePath();
    }

    // ── Animation loop ────────────────────────────────────────────────────────
    const SNAP_MS  = 900;   // ms per epoch: ~300ms fwd + 300ms bwd + 300ms update
    const TOTAL_MS = (snaps.length - 1) * SNAP_MS;
    let startTime  = null;
    let rafId      = null;

    function loop(ts) {
      if (!startTime) startTime = ts;
      const elapsed = Math.min(ts - startTime, TOTAL_MS);
      const si = Math.min(Math.floor(elapsed / SNAP_MS), snaps.length - 2);
      const t  = (elapsed % SNAP_MS) / SNAP_MS;
      drawFrame(si, t);
      if (elapsed < TOTAL_MS) {
        rafId = requestAnimationFrame(loop);
      } else {
        drawFrame(snaps.length - 1, 1);
      }
    }

    rafId = requestAnimationFrame(loop);
    el._cancelAnim = () => { if (rafId) cancelAnimationFrame(rafId); };
  }

  // ── SLIDE: Backprop — Simple 1-neuron case ───────────────────────────────
  function initBackpropSimple(el) {
    el.innerHTML = '';
    const W = 1100, H = 510;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%').style('overflow', 'visible');

    const timers = [];
    el._timers = timers;
    const later = (ms, fn) => timers.push(setTimeout(fn, ms));

    addArrowDef(svg, 'bps-fwd', C.primary);
    addArrowDef(svg, 'bps-bwd', C.accent);
    addArrowDef(svg, 'bps-grn', '#4ade80');

    // \u2500\u2500 layout constants \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    const CY = 190;   // vertical centre of computation chain
    const BW = 80;    // width of rect-style nodes
    const BH = 44;    // height of rect-style nodes
    const SR = 30;    // radius of the \u03a3 circle
    const xC = 115, sC = 320, aC = 530, lC = 760;
    const BY = CY - 100;  // bias y
    const YY = CY - 100;  // true-label y

    // \u2500\u2500 formula box helper \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    function fmtBox(delay, bx, by, bw, header, lines) {
      const PAD = 10, LH = 22;
      const bh = PAD + 14 + lines.length * LH + PAD;
      const g = svg.append('g').attr('opacity', 0);
      g.append('rect').attr('x', bx).attr('y', by).attr('width', bw).attr('height', bh)
        .attr('rx', 6).attr('fill', 'rgba(0,0,0,0.5)')
        .attr('stroke', C.accent).attr('stroke-width', 1.2);
      g.append('text').attr('x', bx + PAD).attr('y', by + PAD + 10)
        .attr('fill', C.muted).attr('font-size', 9).attr('font-family', 'sans-serif')
        .text(header);
      g.append('line').attr('x1', bx + 4).attr('y1', by + PAD + 16)
        .attr('x2', bx + bw - 4).attr('y2', by + PAD + 16)
        .attr('stroke', 'rgba(255,0,110,0.25)').attr('stroke-width', 0.8);
      lines.forEach((ln, i) => {
        g.append('text').attr('x', bx + PAD).attr('y', by + PAD + 16 + (i + 1) * LH)
          .attr('fill', ln.color).attr('font-size', ln.size || 12)
          .attr('font-family', 'monospace').attr('font-weight', ln.bold ? 'bold' : 'normal')
          .text(ln.text);
      });
      g.transition().delay(delay).duration(380).attr('opacity', 1);
      return g;
    }

    // \u2500\u2500 helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    function fg(delay, build) {
      const g = svg.append('g').attr('opacity', 0);
      build(g);
      g.transition().delay(delay).duration(380).attr('opacity', 1);
      return g;
    }

    function fwdArrow(g, x1, x2, y, label, labelDy) {
      g.append('line').attr('x1', x1).attr('y1', y).attr('x2', x2).attr('y2', y)
        .attr('stroke', C.primary).attr('stroke-width', 2).attr('marker-end', 'url(#bps-fwd)');
      if (label) g.append('text').attr('x', (x1 + x2) / 2).attr('y', y + (labelDy || -11))
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 10.5).attr('font-family', 'monospace').text(label);
    }

    function bwdArrow(g, x1, x2, y) {
      g.append('line').attr('x1', x1).attr('y1', y).attr('x2', x2).attr('y2', y)
        .attr('stroke', C.accent).attr('stroke-width', 2.2).attr('marker-end', 'url(#bps-bwd)');
    }

    // \u2500\u2500 phase label (top-centre) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    const phLbl = svg.append('text').attr('x', 550).attr('y', 30)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 16).attr('font-family', 'sans-serif').attr('font-style', 'italic')
      .text('forward pass  \u00b7  x \u2192 \u03a3(wx+b) \u2192 \u03c3(z) \u2192 L(a,y)');

    // \u2500\u2500 reference panel: activation functions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    fg(0, g => {
      const bx = 935, by = 55, bw = 148, bh = 158;
      g.append('rect').attr('x', bx).attr('y', by).attr('width', bw).attr('height', bh)
        .attr('rx', 6).attr('fill', 'rgba(0,0,0,0.4)')
        .attr('stroke', C.muted).attr('stroke-width', 1).attr('stroke-dasharray', '3,2');
      g.append('text').attr('x', bx + 8).attr('y', by + 14)
        .attr('fill', C.muted).attr('font-size', 9).attr('font-family', 'sans-serif')
        .text('Activation Functions');
      g.append('line').attr('x1', bx + 4).attr('y1', by + 20)
        .attr('x2', bx + bw - 4).attr('y2', by + 20)
        .attr('stroke', 'rgba(255,255,255,0.12)').attr('stroke-width', 0.8);
      g.append('text').attr('x', bx + 8).attr('y', by + 34)
        .attr('fill', C.secondary).attr('font-size', 11).attr('font-family', 'monospace')
        .attr('font-weight', 'bold').text('Sigmoid  \u03c3(z)');
      g.append('text').attr('x', bx + 8).attr('y', by + 50)
        .attr('fill', C.muted).attr('font-size', 10).attr('font-family', 'monospace')
        .text('= 1 / (1 + e^\u2212z)');
      g.append('text').attr('x', bx + 8).attr('y', by + 66)
        .attr('fill', C.secondary).attr('font-size', 11).attr('font-family', 'monospace')
        .attr('font-weight', 'bold').text("\u03c3'(z) = a(1\u2212a)");
      g.append('line').attr('x1', bx + 4).attr('y1', by + 78)
        .attr('x2', bx + bw - 4).attr('y2', by + 78)
        .attr('stroke', 'rgba(255,255,255,0.08)').attr('stroke-width', 0.8);
      g.append('text').attr('x', bx + 8).attr('y', by + 94)
        .attr('fill', C.primary).attr('font-size', 11).attr('font-family', 'monospace')
        .attr('font-weight', 'bold').text('ReLU  f(z)');
      g.append('text').attr('x', bx + 8).attr('y', by + 110)
        .attr('fill', C.muted).attr('font-size', 10).attr('font-family', 'monospace')
        .text('= max(0, z)');
      g.append('text').attr('x', bx + 8).attr('y', by + 126)
        .attr('fill', C.primary).attr('font-size', 11).attr('font-family', 'monospace')
        .attr('font-weight', 'bold').text("f'(z) = 1  (z > 0)");
      g.append('text').attr('x', bx + 8).attr('y', by + 142)
        .attr('fill', C.muted).attr('font-size', 10).attr('font-family', 'monospace')
        .text('       = 0  otherwise');
    });

    // \u2550\u2550\u2550\u2550\u2550\u2550 FORWARD PASS \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

    // input box x
    fg(0, g => {
      g.append('rect').attr('x', xC - BW/2).attr('y', CY - BH/2)
        .attr('width', BW).attr('height', BH).attr('rx', 6)
        .attr('fill', 'rgba(0,217,255,0.10)').attr('stroke', C.primary).attr('stroke-width', 1.5);
      g.append('text').attr('x', xC).attr('y', CY - 5)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 15).attr('font-weight', 'bold').attr('font-family', 'monospace').text('x');
      g.append('text').attr('x', xC).attr('y', CY + 13)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 11).attr('font-family', 'monospace').text('= 0.50');
      g.append('text').attr('x', xC).attr('y', CY + BH/2 + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('input');
    });

    // bias circle b (above \u03a3) + dashed arrow
    fg(400, g => {
      g.append('circle').attr('cx', sC).attr('cy', BY).attr('r', 22)
        .attr('fill', 'rgba(255,0,110,0.10)').attr('stroke', C.accent).attr('stroke-width', 1.3);
      g.append('text').attr('x', sC).attr('y', BY - 3)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 13).attr('font-weight', 'bold').attr('font-family', 'monospace').text('b');
      g.append('text').attr('x', sC).attr('y', BY + 13)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 10).attr('font-family', 'monospace').text('= 0.10');
      g.append('line').attr('x1', sC).attr('y1', BY + 22)
        .attr('x2', sC).attr('y2', CY - SR - 2)
        .attr('stroke', C.accent).attr('stroke-width', 1.4).attr('stroke-dasharray', '4,3')
        .attr('marker-end', 'url(#bps-bwd)');
    });

    // w-edge x\u2192\u03a3
    fg(400, g => fwdArrow(g, xC + BW/2, sC - SR - 2, CY, 'w = 0.80'));

    // \u03a3 summing circle
    fg(780, g => {
      g.append('circle').attr('cx', sC).attr('cy', CY).attr('r', SR)
        .attr('fill', 'rgba(157,78,221,0.15)').attr('stroke', C.secondary).attr('stroke-width', 2);
      g.append('text').attr('x', sC).attr('y', CY + 8)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 22).attr('font-family', 'sans-serif').text('\u03a3');
      g.append('text').attr('x', sC).attr('y', CY + SR + 17)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 10).attr('font-family', 'monospace').text('z = wx+b = 0.50');
      g.append('text').attr('x', sC).attr('y', CY - SR - 8)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('linear combination');
    });

    // z-value edge \u03a3\u2192\u03c3
    fg(1160, g => fwdArrow(g, sC + SR + 2, aC - BW/2 - 2, CY, 'z = 0.50'));

    // \u03c3 activation box
    fg(1160, g => {
      g.append('rect').attr('x', aC - BW/2).attr('y', CY - BH/2)
        .attr('width', BW).attr('height', BH).attr('rx', 10)
        .attr('fill', 'rgba(157,78,221,0.15)').attr('stroke', C.secondary).attr('stroke-width', 1.5);
      g.append('text').attr('x', aC).attr('y', CY + 7)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 18).attr('font-family', 'serif').text('\u03c3');
      g.append('text').attr('x', aC).attr('y', CY + BH/2 + 17)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 10).attr('font-family', 'monospace').text('a = \u03c3(z) = 0.622');
      g.append('text').attr('x', aC).attr('y', CY - BH/2 - 8)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('activation');
    });

    // a-value edge \u03c3\u2192L
    fg(1550, g => fwdArrow(g, aC + BW/2 + 2, lC - BW/2 - 2, CY, 'a = 0.622'));

    // true label y (above L box)
    fg(1550, g => {
      g.append('text').attr('x', lC).attr('y', YY - 2)
        .attr('text-anchor', 'middle').attr('fill', '#4ade80')
        .attr('font-size', 13).attr('font-weight', 'bold').attr('font-family', 'monospace').text('y = 1.0');
      g.append('text').attr('x', lC).attr('y', YY + 13)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('true label');
      g.append('line').attr('x1', lC).attr('y1', YY + 20)
        .attr('x2', lC).attr('y2', CY - BH/2 - 2)
        .attr('stroke', '#4ade80').attr('stroke-width', 1.3).attr('stroke-dasharray', '4,3')
        .attr('marker-end', 'url(#bps-grn)');
    });

    // L loss box
    fg(1950, g => {
      g.append('rect').attr('x', lC - BW/2).attr('y', CY - BH/2)
        .attr('width', BW).attr('height', BH).attr('rx', 6)
        .attr('fill', 'rgba(255,0,110,0.12)').attr('stroke', C.accent).attr('stroke-width', 1.5);
      g.append('text').attr('x', lC).attr('y', CY - 5)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 16).attr('font-weight', 'bold').attr('font-family', 'monospace').text('L');
      g.append('text').attr('x', lC).attr('y', CY + 13)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 11).attr('font-family', 'monospace').text('= 0.0714');
      g.append('text').attr('x', lC).attr('y', CY + BH/2 + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('MSE loss');
    });

    // Loss formula to the RIGHT of the L box
    const formX = lC + BW/2 + 14;  // x = 814
    fg(1950, g => {
      g.append('text').attr('x', formX).attr('y', CY - 8)
        .attr('fill', C.accent).attr('font-size', 12).attr('font-family', 'monospace')
        .text('= \u00bd(a \u2212 y)\u00b2');
      g.append('text').attr('x', formX).attr('y', CY + 12)
        .attr('fill', C.muted).attr('font-size', 10).attr('font-family', 'monospace')
        .text('= 0.0714');
    });

    // \u2550\u2550\u2550\u2550\    // \u2550\u2550\u2550\u2550\u2550\u2550 BACKWARD PASS \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    later(3000, () => {
      phLbl.transition().duration(300)
        .attr('fill', C.accent).attr('font-style', 'normal')
        .text('\u2190 backward pass: chain rule  \u2202L/\u2202w = \u2202L/\u2202a \u00b7 \u2202a/\u2202z \u00b7 \u2202z/\u2202w');
    });

    // backward-arrow row sits just below the chain, clear of all nodes
    const bY = CY + SR + 26;

    // backward arrows \u2014 formula boxes appear directly beneath each one
    fg(3000, g => bwdArrow(g, lC - BW/2 - 2, aC + BW/2 + 2, bY));
    fg(3700, g => bwdArrow(g, aC - BW/2 - 2, sC + SR + 2, bY));
    fg(4400, g => bwdArrow(g, sC - SR - 2, xC + BW/2 + 2, bY));

    // \u2500\u2500 formula boxes centered under each backward arrow \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    const fby = bY + 18;  // top of formula boxes = 264

    // arrow 1 midX = (718+572)/2 = 645  \u2192  box: x=555, w=180
    fmtBox(3000, 555, fby, 180, 'L \u2192 \u03c3  (output gradient)', [
      { text: '\u2202L/\u2202a = a \u2212 y', color: C.accent, size: 13, bold: true },
      { text: '= 0.622 \u2212 1.0 = \u22120.378', color: C.muted, size: 11 },
    ]);

    // arrow 2 midX = (488+352)/2 = 420  \u2192  box: x=320, w=200
    fmtBox(3700, 320, fby, 200, '\u03c3 \u2192 \u03a3  (activation gradient)', [
      { text: "\u2202a/\u2202z = \u03c3'(z) = a(1\u2212a)", color: C.secondary, size: 12, bold: true },
      { text: '= 0.622 \u00d7 0.378', color: C.muted, size: 11 },
      { text: '= 0.235', color: C.secondary, size: 12, bold: true },
    ]);

    // arrow 3 midX = (288+157)/2 = 222  \u2192  box: x=144, w=156
    fmtBox(4400, 144, fby, 156, '\u03a3 \u2192 x  (weight gradient)', [
      { text: '\u2202z/\u2202w = x = 0.50', color: C.primary, size: 12, bold: true },
    ]);

    // \u2500\u2500 chain result banner \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    later(5100, () => {
      phLbl.transition().duration(300)
        .attr('fill', '#4ade80')
        .text('gradient computed \u2014 apply update  w \u2190 w \u2212 \u03b7 \u00b7 \u2202L/\u2202w');
    });

    fg(5100, g => {
      const by = fby + 120;  // 264 + 120 = 384
      g.append('rect').attr('x', 80).attr('y', by).attr('width', 840).attr('height', 76)
        .attr('rx', 6).attr('fill', 'rgba(0,217,255,0.05)')
        .attr('stroke', 'rgba(0,217,255,0.2)').attr('stroke-width', 1);
      g.append('text').attr('x', 460).attr('y', by + 26)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 12).attr('font-family', 'monospace')
        .text('\u2202L/\u2202w  =  \u2202L/\u2202a \u00b7 \u2202a/\u2202z \u00b7 \u2202z/\u2202w  =  (\u22120.378)(0.235)(0.50)  =  \u22120.044');
      g.append('text').attr('x', 460).attr('y', by + 52)
        .attr('text-anchor', 'middle').attr('fill', '#4ade80')
        .attr('font-size', 12).attr('font-family', 'monospace')
        .text('w  \u2190  0.800 \u2212 0.1\u00d7(\u22120.044)  =  0.804  \u2713');
    });
  }
  // ── SLIDE: Backprop — Multi-layer network gradient flow ──────────────────
  function initBackpropNetwork(el) {
    el.innerHTML = '';
    const W = 1100, H = 510;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%').style('overflow', 'visible');

    const timers = [];
    el._timers = timers;
    const later = (ms, fn) => timers.push(setTimeout(fn, ms));

    addArrowDef(svg, 'bpn-fwd', C.primary);
    addArrowDef(svg, 'bpn-bwd', C.accent);
    addArrowDef(svg, 'bpn-grn', '#4ade80');

    // ── layout ──────────────────────────────────────────────────────────────
    const BW = 68, BH = 38, SR = 22, GR = 10;
    const xN = 70, s1N = 215, a1N = 355, s2N = 505, a2N = 645, lN = 790;

    // 4-row grid — highlighted path zigzags: input row2 → neuron1 row0 → neuron2 row3
    const rowY  = [75, 140, 205, 270];
    const hiIn  = 2;   // y=205
    const hiR1  = 0;   // y=75
    const hiR2  = 3;   // y=270
    const lossY = Math.round((rowY[0] + rowY[rowY.length - 1]) / 2); // 173

    const hiInY = rowY[hiIn];   // 205
    const hi1Y  = rowY[hiR1];   // 75
    const hi2Y  = rowY[hiR2];   // 270

    // bias circles placed to avoid grid-node overlaps
    const b1Cx = s1N - 62, b1Cy = hi1Y + 30;   // (153, 105)  below-left of Σ₁
    const b2Cx = s2N - 62, b2Cy = hi2Y - 42;   // (443, 228)  above-left of Σ₂

    // backward pass horizontal bar below network
    const bY  = hi2Y + SR + 22;   // 314
    const fby = bY + 18;          // 332

    // ── helpers ─────────────────────────────────────────────────────────────
    function fmtBox(delay, bx, by, bw, header, lines) {
      const PAD = 10, LH = 22;
      const bh = PAD + 14 + lines.length * LH + PAD;
      const g = svg.append('g').attr('opacity', 0);
      g.append('rect').attr('x', bx).attr('y', by).attr('width', bw).attr('height', bh)
        .attr('rx', 6).attr('fill', 'rgba(0,0,0,0.55)')
        .attr('stroke', C.accent).attr('stroke-width', 1.2);
      g.append('text').attr('x', bx + PAD).attr('y', by + PAD + 10)
        .attr('fill', C.muted).attr('font-size', 9).attr('font-family', 'sans-serif')
        .text(header);
      g.append('line').attr('x1', bx + 4).attr('y1', by + PAD + 16)
        .attr('x2', bx + bw - 4).attr('y2', by + PAD + 16)
        .attr('stroke', 'rgba(255,0,110,0.25)').attr('stroke-width', 0.8);
      lines.forEach((ln, i) => {
        g.append('text').attr('x', bx + PAD).attr('y', by + PAD + 16 + (i + 1) * LH)
          .attr('fill', ln.color).attr('font-size', ln.size || 12)
          .attr('font-family', 'monospace').attr('font-weight', ln.bold ? 'bold' : 'normal')
          .text(ln.text);
      });
      g.transition().delay(delay).duration(380).attr('opacity', 1);
      return g;
    }

    function fg(delay, build) {
      const g = svg.append('g').attr('opacity', 0);
      build(g);
      g.transition().delay(delay).duration(380).attr('opacity', 1);
      return g;
    }

    // diagonal-capable forward arrow; label placed above midpoint
    function fwdArrow(g, x1, y1, x2, y2, label) {
      g.append('line')
        .attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2)
        .attr('stroke', C.primary).attr('stroke-width', 2)
        .attr('marker-end', 'url(#bpn-fwd)');
      if (label)
        g.append('text').attr('x', (x1 + x2) / 2).attr('y', (y1 + y2) / 2 - 12)
          .attr('text-anchor', 'middle').attr('fill', C.primary)
          .attr('font-size', 10).attr('font-family', 'monospace').text(label);
    }

    function bwdArrow(g, x1, x2, y) {
      g.append('line').attr('x1', x1).attr('y1', y).attr('x2', x2).attr('y2', y)
        .attr('stroke', C.accent).attr('stroke-width', 2.2)
        .attr('marker-end', 'url(#bpn-bwd)');
    }

    // ── phase label ─────────────────────────────────────────────────────────
    const phLbl = svg.append('text').attr('x', 430).attr('y', 30)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 16).attr('font-family', 'sans-serif').attr('font-style', 'italic')
      .text('2 neurons chained \u00b7 same forward rule applied twice');

    // ── annotation panel (right side, always visible) ────────────────────────
    svg.append('g').call(g => {
      const bx = 862, by = 48, bw = 222, bh = 172;
      g.append('rect').attr('x', bx).attr('y', by).attr('width', bw).attr('height', bh)
        .attr('rx', 6).attr('fill', 'rgba(0,0,0,0.40)')
        .attr('stroke', 'rgba(255,255,255,0.12)').attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,3');
      [
        [by + 18,  C.muted,   9.5, false, 'the neuron gets a'],
        [by + 33,  C.muted,   9.5, false, 'single error signal'],
        [by + 57,  C.primary, 9.5, false, 'backprop asks: how much'],
        [by + 72,  C.primary, 9.5, false, 'did each weight'],
        [by + 87,  C.primary, 9.5, false, 'contribute to that error?'],
        [by + 111, C.accent,  9.5, false, 'each weight is blamed'],
        [by + 126, C.accent,  9.5, false, 'by its actual effect \u2014'],
        [by + 148, C.accent,  11,  true,  'not evenly'],
      ].forEach(([y, color, size, bold, text]) =>
        g.append('text').attr('x', bx + 12).attr('y', y)
          .attr('fill', color).attr('font-size', size)
          .attr('font-family', 'sans-serif').attr('font-weight', bold ? 'bold' : 'normal')
          .text(text));
    });

    // ── ghost grid: full network breadth ────────────────────────────────────
    svg.append('g').call(g => {
      const GC = 'rgba(157,78,221,0.70)';
      const GF = 'rgba(157,78,221,0.18)';
      const GL = 'rgba(0,217,255,0.30)';
      const DA = '3,3';
      const DL = '2,3';

      // Full connectivity: each input → each \u03c3\u2081 (neuron 1 outputs)
      rowY.forEach(iy => rowY.forEach(a1y =>
        g.append('line')
          .attr('x1', xN + BW / 2).attr('y1', iy)
          .attr('x2', a1N - GR - 1).attr('y2', a1y)
          .attr('stroke', GL).attr('stroke-width', 0.9).attr('stroke-dasharray', DL)));

      // Full connectivity: each \u03c3\u2081 → each \u03c3\u2082 (neuron 2 outputs)
      rowY.forEach(a1y => rowY.forEach(a2y =>
        g.append('line')
          .attr('x1', a1N + GR + 1).attr('y1', a1y)
          .attr('x2', a2N - GR - 1).attr('y2', a2y)
          .attr('stroke', GL).attr('stroke-width', 0.9).attr('stroke-dasharray', DL)));

      // All \u03c3\u2082 converge to single loss node
      rowY.forEach(a2y =>
        g.append('line')
          .attr('x1', a2N + GR + 1).attr('y1', a2y)
          .attr('x2', lN - BW / 2 - 1).attr('y2', lossY)
          .attr('stroke', GL).attr('stroke-width', 0.9).attr('stroke-dasharray', DL));

      // Ghost input nodes (skip hiIn row)
      rowY.forEach((ry, i) => {
        if (i === hiIn) return;
        g.append('rect')
          .attr('x', xN - BW / 2).attr('y', ry - BH / 2)
          .attr('width', BW).attr('height', BH).attr('rx', 4)
          .attr('fill', 'rgba(0,217,255,0.10)')
          .attr('stroke', 'rgba(0,217,255,0.45)').attr('stroke-width', 1.2)
          .attr('stroke-dasharray', DA);
      });

      // Ghost \u03c3\u2081 circles only (activation column, skip hiR1)
      rowY.forEach((ry, i) => {
        if (i === hiR1) return;
        g.append('circle').attr('cx', a1N).attr('cy', ry).attr('r', GR)
          .attr('fill', GF).attr('stroke', GC).attr('stroke-width', 1.5)
          .attr('stroke-dasharray', DA);
      });

      // Ghost \u03c3\u2082 circles only (activation column, skip hiR2)
      rowY.forEach((ry, i) => {
        if (i === hiR2) return;
        g.append('circle').attr('cx', a2N).attr('cy', ry).attr('r', GR)
          .attr('fill', GF).attr('stroke', GC).attr('stroke-width', 1.5)
          .attr('stroke-dasharray', DA);
      });

      // Ghost loss (single node at lossY)
      g.append('rect')
        .attr('x', lN - BW / 2).attr('y', lossY - BH / 2)
        .attr('width', BW).attr('height', BH).attr('rx', 6)
        .attr('fill', 'rgba(74,222,128,0.06)')
        .attr('stroke', 'rgba(74,222,128,0.25)').attr('stroke-width', 1)
        .attr('stroke-dasharray', DA);
    });

    // \u2550\u2550\u2550\u2550\u2550\u2550 FORWARD PASS \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

    // Highlighted input (row hiIn, y=205)
    fg(0, g => {
      g.append('rect').attr('x', xN - BW / 2).attr('y', hiInY - BH / 2)
        .attr('width', BW).attr('height', BH).attr('rx', 6)
        .attr('fill', 'rgba(0,217,255,0.10)').attr('stroke', C.primary).attr('stroke-width', 1.5);
      g.append('text').attr('x', xN).attr('y', hiInY - 4)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 14).attr('font-weight', 'bold').attr('font-family', 'monospace').text('x');
      g.append('text').attr('x', xN).attr('y', hiInY + 12)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9.5).attr('font-family', 'monospace').text('= 0.50');
      g.append('text').attr('x', xN).attr('y', hiInY + BH / 2 + 13)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8.5).attr('font-family', 'sans-serif').text('input');
    });

    // Bias b\u2081 (below-left of \u03a3\u2081 to avoid ghost at row above)
    fg(300, g => {
      g.append('circle').attr('cx', b1Cx).attr('cy', b1Cy).attr('r', 18)
        .attr('fill', 'rgba(255,0,110,0.10)').attr('stroke', C.accent).attr('stroke-width', 1.3);
      g.append('text').attr('x', b1Cx).attr('y', b1Cy - 1)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 10).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('b\u2081');
      g.append('text').attr('x', b1Cx).attr('y', b1Cy + 11)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'monospace').text('0.10');
      g.append('line')
        .attr('x1', b1Cx + 14).attr('y1', b1Cy - 10)
        .attr('x2', s1N - SR - 2).attr('y2', hi1Y + SR / 2)
        .attr('stroke', C.accent).attr('stroke-width', 1.2).attr('stroke-dasharray', '4,3')
        .attr('marker-end', 'url(#bpn-bwd)');
    });

    // w\u2081: diagonal arrow input(y=205) \u2192 \u03a3\u2081(y=75) — goes up through grid
    fg(300, g => fwdArrow(g, xN + BW / 2, hiInY, s1N - SR - 2, hi1Y, 'w\u2081=0.80'));

    // \u03a3\u2081 — pre-activation of neuron 1 (row 0, top of grid)
    fg(600, g => {
      g.append('circle').attr('cx', s1N).attr('cy', hi1Y).attr('r', SR)
        .attr('fill', 'rgba(157,78,221,0.18)').attr('stroke', C.secondary).attr('stroke-width', 2.2);
      g.append('text').attr('x', s1N).attr('y', hi1Y + 7)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 18).attr('font-family', 'sans-serif').text('\u03a3');
      g.append('text').attr('x', s1N).attr('y', hi1Y + SR + 14)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 9).attr('font-family', 'monospace').text('z\u2081=0.50');
      g.append('text').attr('x', s1N).attr('y', hi1Y - SR - 7)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'sans-serif').text('pre-act');
    });

    // z\u2081 horizontal: \u03a3\u2081 \u2192 \u03c3\u2081
    fg(900, g => fwdArrow(g, s1N + SR + 2, hi1Y, a1N - BW / 2 - 2, hi1Y, 'z\u2081=0.50'));

    // \u03c3\u2081 — activation of neuron 1
    fg(900, g => {
      g.append('rect').attr('x', a1N - BW / 2).attr('y', hi1Y - BH / 2)
        .attr('width', BW).attr('height', BH).attr('rx', 10)
        .attr('fill', 'rgba(157,78,221,0.18)').attr('stroke', C.secondary).attr('stroke-width', 2);
      g.append('text').attr('x', a1N).attr('y', hi1Y + 6)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 13).attr('font-family', 'serif').text('\u03c3\u2081');
      g.append('text').attr('x', a1N).attr('y', hi1Y + BH / 2 + 13)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 9).attr('font-family', 'monospace').text('a\u2081=0.622');
      g.append('text').attr('x', a1N).attr('y', hi1Y - BH / 2 - 7)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'sans-serif').text('act  \u2014  neuron 1');
    });

    // Bias b\u2082 (above-left of \u03a3\u2082)
    fg(1200, g => {
      g.append('circle').attr('cx', b2Cx).attr('cy', b2Cy).attr('r', 18)
        .attr('fill', 'rgba(255,0,110,0.10)').attr('stroke', C.accent).attr('stroke-width', 1.3);
      g.append('text').attr('x', b2Cx).attr('y', b2Cy - 1)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 10).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('b\u2082');
      g.append('text').attr('x', b2Cx).attr('y', b2Cy + 11)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'monospace').text('0.05');
      g.append('line')
        .attr('x1', b2Cx + 14).attr('y1', b2Cy + 10)
        .attr('x2', s2N - SR - 2).attr('y2', hi2Y - SR / 2)
        .attr('stroke', C.accent).attr('stroke-width', 1.2).attr('stroke-dasharray', '4,3')
        .attr('marker-end', 'url(#bpn-bwd)');
    });

    // w\u2082: diagonal arrow \u03c3\u2081(y=75) \u2192 \u03a3\u2082(y=270) — goes down through grid
    fg(1200, g => fwdArrow(g, a1N + BW / 2 + 2, hi1Y, s2N - SR - 2, hi2Y, 'w\u2082=0.60'));

    // \u03a3\u2082 — pre-activation of neuron 2 (row 3, bottom of grid)
    fg(1500, g => {
      g.append('circle').attr('cx', s2N).attr('cy', hi2Y).attr('r', SR)
        .attr('fill', 'rgba(157,78,221,0.18)').attr('stroke', C.secondary).attr('stroke-width', 2.2);
      g.append('text').attr('x', s2N).attr('y', hi2Y + 7)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 18).attr('font-family', 'sans-serif').text('\u03a3');
      g.append('text').attr('x', s2N).attr('y', hi2Y + SR + 14)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 9).attr('font-family', 'monospace').text('z\u2082=0.47');
      g.append('text').attr('x', s2N).attr('y', hi2Y - SR - 7)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'sans-serif').text('pre-act');
    });

    // z\u2082 horizontal: \u03a3\u2082 \u2192 \u03c3\u2082
    fg(1800, g => fwdArrow(g, s2N + SR + 2, hi2Y, a2N - BW / 2 - 2, hi2Y, 'z\u2082=0.47'));

    // \u03c3\u2082 — activation of neuron 2
    fg(1800, g => {
      g.append('rect').attr('x', a2N - BW / 2).attr('y', hi2Y - BH / 2)
        .attr('width', BW).attr('height', BH).attr('rx', 10)
        .attr('fill', 'rgba(157,78,221,0.18)').attr('stroke', C.secondary).attr('stroke-width', 2);
      g.append('text').attr('x', a2N).attr('y', hi2Y + 6)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 13).attr('font-family', 'serif').text('\u03c3\u2082');
      g.append('text').attr('x', a2N).attr('y', hi2Y + BH / 2 + 13)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 9).attr('font-family', 'monospace').text('a\u2082=0.615');
      g.append('text').attr('x', a2N).attr('y', hi2Y - BH / 2 - 7)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'sans-serif').text('act  \u2014  neuron 2');
    });

    // \u03c3\u2082 \u2192 L (diagonal up to center loss)
    fg(2100, g => {
      g.append('line').attr('x1', a2N + BW / 2 + 2).attr('y1', hi2Y)
        .attr('x2', lN - BW / 2 - 2).attr('y2', lossY)
        .attr('stroke', C.primary).attr('stroke-width', 2)
        .attr('marker-end', 'url(#bpn-fwd)');
    });

    // Loss node (center of grid, all outputs converge here)
    fg(2100, g => {
      g.append('rect').attr('x', lN - BW / 2).attr('y', lossY - BH / 2)
        .attr('width', BW).attr('height', BH).attr('rx', 6)
        .attr('fill', 'rgba(74,222,128,0.15)').attr('stroke', '#4ade80').attr('stroke-width', 2);
      g.append('text').attr('x', lN).attr('y', lossY + 7)
        .attr('text-anchor', 'middle').attr('fill', '#4ade80')
        .attr('font-size', 14).attr('font-weight', 'bold').attr('font-family', 'monospace').text('L');
      g.append('text').attr('x', lN).attr('y', lossY + BH / 2 + 13)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'monospace').text('y=0.0  L=0.912');
      g.append('text').attr('x', lN).attr('y', lossY - BH / 2 - 7)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8.5).attr('font-family', 'sans-serif').text('loss');
    });

    // \u2550\u2550\u2550\u2550\u2550\u2550 BACKWARD PASS \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    later(3200, () => {
      phLbl.transition().duration(300)
        .attr('fill', C.accent).attr('font-style', 'normal')
        .text('\u2190 same chain rule as before \u2014 starting at the output neuron');
    });

    fg(3200, g => bwdArrow(g, lN - BW / 2 - 2, a2N + BW / 2 + 2, bY));
    fg(3900, g => bwdArrow(g, a2N - BW / 2 - 2, s2N + SR + 2, bY));

    fmtBox(3200, 545, fby, 200, 'output neuron  (same as 1-neuron slide)', [
      { text: '\u2202L/\u2202a\u2082 = a\u2082\u2212y = \u22120.397', color: C.accent, size: 12, bold: true },
      { text: "\u2202a\u2082/\u2202z\u2082 = \u03c3'(z\u2082) = 0.239", color: C.secondary, size: 11 },
      { text: '\u2202L/\u2202w\u2082 = \u22120.059  \u2713', color: '#4ade80', size: 12, bold: true },
    ]);

    later(4700, () => {
      phLbl.transition().duration(300)
        .attr('fill', C.secondary)
        .text('\u2190 gradient crosses into hidden neuron through w\u2082');
    });

    fg(4700, g => bwdArrow(g, s2N - SR - 2, a1N + BW / 2 + 2, bY));

    fmtBox(4700, 348, fby, 175, 'chain to hidden neuron', [
      { text: '\u2202L/\u2202a\u2081 = \u2202L/\u2202z\u2082 \u00b7 w\u2082', color: C.secondary, size: 11, bold: true },
      { text: '= (\u22120.397\u00d70.239) \u00d7 0.60', color: C.muted, size: 10 },
      { text: '= \u22120.057', color: C.secondary, size: 12, bold: true },
    ]);

    fg(5600, g => bwdArrow(g, a1N - BW / 2 - 2, s1N + SR + 2, bY));
    fg(6200, g => bwdArrow(g, s1N - SR - 2, xN + BW / 2 + 2, bY));

    fmtBox(5600, 114, fby, 200, 'hidden neuron  (same rule again)', [
      { text: "\u2202a\u2081/\u2202z\u2081 = \u03c3'(z\u2081) = 0.235", color: C.secondary, size: 11 },
      { text: '\u2202L/\u2202w\u2081 = \u22120.0067  \u2713', color: '#4ade80', size: 12, bold: true },
    ]);

    later(7000, () => {
      phLbl.transition().duration(300).attr('fill', '#4ade80')
        .text('both weights updated \u2014 same 3-step rule at every layer');
    });

    fg(7000, g => {
      const by = fby + 110;
      g.append('rect').attr('x', 80).attr('y', by).attr('width', 760).attr('height', 56)
        .attr('rx', 6).attr('fill', 'rgba(0,217,255,0.05)')
        .attr('stroke', 'rgba(0,217,255,0.2)').attr('stroke-width', 1);
      g.append('text').attr('x', 420).attr('y', by + 22)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 12).attr('font-family', 'monospace').attr('font-weight', 'bold')
        .text('w\u2082  \u2190  0.600 \u2212 0.1\u00d7(\u22120.059)  =  0.606  \u2713');
      g.append('text').attr('x', 420).attr('y', by + 44)
        .attr('text-anchor', 'middle').attr('fill', '#4ade80')
        .attr('font-size', 12).attr('font-family', 'monospace').attr('font-weight', 'bold')
        .text('w\u2081  \u2190  0.800 \u2212 0.1\u00d7(\u22120.0067)  =  0.801  \u2713');
    });
  }


  // ── SLIDE: Bias backpropagation — same chain rule, ∂z/∂b = 1 ────────────
  function initBiasBackprop(el) {
    el.innerHTML = '';
    const W = 1100, H = 510;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%').style('overflow', 'visible');

    const timers = [];
    el._timers = timers;
    const later = (ms, fn) => timers.push(setTimeout(fn, ms));

    addArrowDef(svg, 'bpb-fwd', C.primary);
    addArrowDef(svg, 'bpb-bwd', C.accent);
    addArrowDef(svg, 'bpb-grn', '#4ade80');

    // \u2500\u2500 layout constants \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    const CY = 190, BW = 80, BH = 44, SR = 30;
    const xC = 115, sC = 320, aC = 530, lC = 760;
    const BY = CY - 100;  // bias node y = 90
    const YY = CY - 100;  // true-label y

    // \u2500\u2500 formula box helper \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    function fmtBox(delay, bx, by, bw, header, lines) {
      const PAD = 10, LH = 22;
      const bh = PAD + 14 + lines.length * LH + PAD;
      const g = svg.append('g').attr('opacity', 0);
      g.append('rect').attr('x', bx).attr('y', by).attr('width', bw).attr('height', bh)
        .attr('rx', 6).attr('fill', 'rgba(0,0,0,0.5)')
        .attr('stroke', C.accent).attr('stroke-width', 1.2);
      g.append('text').attr('x', bx + PAD).attr('y', by + PAD + 10)
        .attr('fill', C.muted).attr('font-size', 9).attr('font-family', 'sans-serif')
        .text(header);
      g.append('line').attr('x1', bx + 4).attr('y1', by + PAD + 16)
        .attr('x2', bx + bw - 4).attr('y2', by + PAD + 16)
        .attr('stroke', 'rgba(255,0,110,0.25)').attr('stroke-width', 0.8);
      lines.forEach((ln, i) => {
        g.append('text').attr('x', bx + PAD).attr('y', by + PAD + 16 + (i + 1) * LH)
          .attr('fill', ln.color).attr('font-size', ln.size || 12)
          .attr('font-family', 'monospace').attr('font-weight', ln.bold ? 'bold' : 'normal')
          .text(ln.text);
      });
      g.transition().delay(delay).duration(380).attr('opacity', 1);
      return g;
    }

    // \u2500\u2500 helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    function fg(delay, build) {
      const g = svg.append('g').attr('opacity', 0);
      build(g);
      g.transition().delay(delay).duration(380).attr('opacity', 1);
      return g;
    }

    function fwdArrow(g, x1, x2, y, label) {
      g.append('line').attr('x1', x1).attr('y1', y).attr('x2', x2).attr('y2', y)
        .attr('stroke', C.primary).attr('stroke-width', 2).attr('marker-end', 'url(#bpb-fwd)');
      if (label) g.append('text').attr('x', (x1 + x2) / 2).attr('y', y - 11)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 10.5).attr('font-family', 'monospace').text(label);
    }

    function bwdArrow(g, x1, x2, y) {
      g.append('line').attr('x1', x1).attr('y1', y).attr('x2', x2).attr('y2', y)
        .attr('stroke', C.accent).attr('stroke-width', 2.2).attr('marker-end', 'url(#bpb-bwd)');
    }

    // \u2500\u2500 phase label \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    const phLbl = svg.append('text').attr('x', 550).attr('y', 30)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 16).attr('font-family', 'sans-serif').attr('font-style', 'italic')
      .text('forward pass  \u00b7  x \u2192 \u03a3(wx+b) \u2192 \u03c3(z) \u2192 L(a,y)');

    // \u2500\u2500 reference panel: activation functions \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    fg(0, g => {
      const bx = 935, by = 55, bw = 148, bh = 158;
      g.append('rect').attr('x', bx).attr('y', by).attr('width', bw).attr('height', bh)
        .attr('rx', 6).attr('fill', 'rgba(0,0,0,0.4)')
        .attr('stroke', C.muted).attr('stroke-width', 1).attr('stroke-dasharray', '3,2');
      g.append('text').attr('x', bx + 8).attr('y', by + 14)
        .attr('fill', C.muted).attr('font-size', 9).attr('font-family', 'sans-serif')
        .text('Activation Functions');
      g.append('line').attr('x1', bx + 4).attr('y1', by + 20)
        .attr('x2', bx + bw - 4).attr('y2', by + 20)
        .attr('stroke', 'rgba(255,255,255,0.12)').attr('stroke-width', 0.8);
      g.append('text').attr('x', bx + 8).attr('y', by + 34)
        .attr('fill', C.secondary).attr('font-size', 11).attr('font-family', 'monospace')
        .attr('font-weight', 'bold').text('Sigmoid  \u03c3(z)');
      g.append('text').attr('x', bx + 8).attr('y', by + 50)
        .attr('fill', C.muted).attr('font-size', 10).attr('font-family', 'monospace')
        .text('= 1 / (1 + e^\u2212z)');
      g.append('text').attr('x', bx + 8).attr('y', by + 66)
        .attr('fill', C.secondary).attr('font-size', 11).attr('font-family', 'monospace')
        .attr('font-weight', 'bold').text("\u03c3'(z) = a(1\u2212a)");
      g.append('line').attr('x1', bx + 4).attr('y1', by + 78)
        .attr('x2', bx + bw - 4).attr('y2', by + 78)
        .attr('stroke', 'rgba(255,255,255,0.08)').attr('stroke-width', 0.8);
      g.append('text').attr('x', bx + 8).attr('y', by + 94)
        .attr('fill', C.primary).attr('font-size', 11).attr('font-family', 'monospace')
        .attr('font-weight', 'bold').text('ReLU  f(z)');
      g.append('text').attr('x', bx + 8).attr('y', by + 110)
        .attr('fill', C.muted).attr('font-size', 10).attr('font-family', 'monospace')
        .text('= max(0, z)');
      g.append('text').attr('x', bx + 8).attr('y', by + 126)
        .attr('fill', C.primary).attr('font-size', 11).attr('font-family', 'monospace')
        .attr('font-weight', 'bold').text("f'(z) = 1  (z > 0)");
      g.append('text').attr('x', bx + 8).attr('y', by + 142)
        .attr('fill', C.muted).attr('font-size', 10).attr('font-family', 'monospace')
        .text('       = 0  otherwise');
    });

    // \u2550\u2550\u2550\u2550\u2550\u2550 FORWARD PASS \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

    fg(0, g => {
      g.append('rect').attr('x', xC - BW/2).attr('y', CY - BH/2)
        .attr('width', BW).attr('height', BH).attr('rx', 6)
        .attr('fill', 'rgba(0,217,255,0.10)').attr('stroke', C.primary).attr('stroke-width', 1.5);
      g.append('text').attr('x', xC).attr('y', CY - 5)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 15).attr('font-weight', 'bold').attr('font-family', 'monospace').text('x');
      g.append('text').attr('x', xC).attr('y', CY + 13)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 11).attr('font-family', 'monospace').text('= 0.50');
      g.append('text').attr('x', xC).attr('y', CY + BH/2 + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('input');
    });

    // bias node b (highlighted — this is the focus of the slide)
    fg(400, g => {
      g.append('circle').attr('cx', sC).attr('cy', BY).attr('r', 22)
        .attr('fill', 'rgba(255,0,110,0.18)').attr('stroke', C.accent).attr('stroke-width', 2);
      g.append('text').attr('x', sC).attr('y', BY - 3)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 13).attr('font-weight', 'bold').attr('font-family', 'monospace').text('b');
      g.append('text').attr('x', sC).attr('y', BY + 13)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 10).attr('font-family', 'monospace').text('= 0.10');
      g.append('line').attr('x1', sC).attr('y1', BY + 22)
        .attr('x2', sC).attr('y2', CY - SR - 2)
        .attr('stroke', C.accent).attr('stroke-width', 1.4).attr('stroke-dasharray', '4,3')
        .attr('marker-end', 'url(#bpb-fwd)');
    });

    fg(400, g => fwdArrow(g, xC + BW/2, sC - SR - 2, CY, 'w = 0.80'));

    fg(780, g => {
      g.append('circle').attr('cx', sC).attr('cy', CY).attr('r', SR)
        .attr('fill', 'rgba(157,78,221,0.15)').attr('stroke', C.secondary).attr('stroke-width', 2);
      g.append('text').attr('x', sC).attr('y', CY + 8)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 22).attr('font-family', 'sans-serif').text('\u03a3');
      g.append('text').attr('x', sC).attr('y', CY + SR + 17)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 10).attr('font-family', 'monospace').text('z = wx+b = 0.50');
      g.append('text').attr('x', sC).attr('y', CY - SR - 8)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('linear combination');
    });

    fg(1160, g => fwdArrow(g, sC + SR + 2, aC - BW/2 - 2, CY, 'z = 0.50'));

    fg(1160, g => {
      g.append('rect').attr('x', aC - BW/2).attr('y', CY - BH/2)
        .attr('width', BW).attr('height', BH).attr('rx', 10)
        .attr('fill', 'rgba(157,78,221,0.15)').attr('stroke', C.secondary).attr('stroke-width', 1.5);
      g.append('text').attr('x', aC).attr('y', CY + 7)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 18).attr('font-family', 'serif').text('\u03c3');
      g.append('text').attr('x', aC).attr('y', CY + BH/2 + 17)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 10).attr('font-family', 'monospace').text('a = \u03c3(z) = 0.622');
      g.append('text').attr('x', aC).attr('y', CY - BH/2 - 8)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('activation');
    });

    fg(1550, g => fwdArrow(g, aC + BW/2 + 2, lC - BW/2 - 2, CY, 'a = 0.622'));

    fg(1550, g => {
      g.append('text').attr('x', lC).attr('y', YY - 2)
        .attr('text-anchor', 'middle').attr('fill', '#4ade80')
        .attr('font-size', 13).attr('font-weight', 'bold').attr('font-family', 'monospace').text('y = 1.0');
      g.append('text').attr('x', lC).attr('y', YY + 13)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('true label');
      g.append('line').attr('x1', lC).attr('y1', YY + 20)
        .attr('x2', lC).attr('y2', CY - BH/2 - 2)
        .attr('stroke', '#4ade80').attr('stroke-width', 1.3).attr('stroke-dasharray', '4,3')
        .attr('marker-end', 'url(#bpb-grn)');
    });

    fg(1950, g => {
      g.append('rect').attr('x', lC - BW/2).attr('y', CY - BH/2)
        .attr('width', BW).attr('height', BH).attr('rx', 6)
        .attr('fill', 'rgba(255,0,110,0.12)').attr('stroke', C.accent).attr('stroke-width', 1.5);
      g.append('text').attr('x', lC).attr('y', CY - 5)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 16).attr('font-weight', 'bold').attr('font-family', 'monospace').text('L');
      g.append('text').attr('x', lC).attr('y', CY + 13)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 11).attr('font-family', 'monospace').text('= 0.0714');
      g.append('text').attr('x', lC).attr('y', CY + BH/2 + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('MSE loss');
    });

    fg(1950, g => {
      const formX = lC + BW/2 + 14;
      g.append('text').attr('x', formX).attr('y', CY - 8)
        .attr('fill', C.accent).attr('font-size', 12).attr('font-family', 'monospace')
        .text('= \u00bd(a \u2212 y)\u00b2');
      g.append('text').attr('x', formX).attr('y', CY + 12)
        .attr('fill', C.muted).attr('font-size', 10).attr('font-family', 'monospace')
        .text('= 0.0714');
    });

    // \u2550\u2550\u2550\u2550\u2550\u2550 BACKWARD PASS \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    later(3000, () => {
      phLbl.transition().duration(300)
        .attr('fill', C.accent).attr('font-style', 'normal')
        .text('\u2190 backward pass: chain rule  \u2202L/\u2202b = \u2202L/\u2202a \u00b7 \u2202a/\u2202z \u00b7 \u2202z/\u2202b');
    });

    const bY = CY + SR + 26;  // horizontal backward-arrow row = 246
    const fby = bY + 18;      // formula box tops = 264

    // arrows 1 & 2: same horizontal backward flow as weight slide
    fg(3000, g => bwdArrow(g, lC - BW/2 - 2, aC + BW/2 + 2, bY));
    fg(3700, g => bwdArrow(g, aC - BW/2 - 2, sC + SR + 2, bY));

    // arrow 3: vertical — gradient flows UP from \u03a3 to b
    fg(4400, g => {
      g.append('line')
        .attr('x1', sC).attr('y1', CY - SR - 2)
        .attr('x2', sC).attr('y2', BY + 24)
        .attr('stroke', C.accent).attr('stroke-width', 2.2)
        .attr('marker-end', 'url(#bpb-bwd)');
    });

    // formula boxes under horizontal arrows (same positions as weight slide)
    fmtBox(3000, 555, fby, 180, 'L \u2192 \u03c3  (output gradient)', [
      { text: '\u2202L/\u2202a = a \u2212 y', color: C.accent, size: 13, bold: true },
      { text: '= 0.622 \u2212 1.0 = \u22120.378', color: C.muted, size: 11 },
    ]);

    fmtBox(3700, 320, fby, 200, '\u03c3 \u2192 \u03a3  (activation gradient)', [
      { text: "\u2202a/\u2202z = \u03c3'(z) = a(1\u2212a)", color: C.secondary, size: 12, bold: true },
      { text: '= 0.622 \u00d7 0.378 = 0.235', color: C.muted, size: 11 },
    ]);

    // formula box beside b — the key insight
    fmtBox(4400, 355, 50, 175, '\u03a3 \u2192 b  (bias gradient)', [
      { text: '\u2202z/\u2202b = 1', color: C.accent, size: 13, bold: true },
      { text: 'b has no input \u2014 always 1', color: C.muted, size: 10 },
    ]);

    // chain result banner
    later(5100, () => {
      phLbl.transition().duration(300)
        .attr('fill', '#4ade80')
        .text('gradient computed \u2014 apply update  b \u2190 b \u2212 \u03b7 \u00b7 \u2202L/\u2202b');
    });

    fg(5100, g => {
      const by = fby + 120;
      g.append('rect').attr('x', 80).attr('y', by).attr('width', 840).attr('height', 76)
        .attr('rx', 6).attr('fill', 'rgba(0,217,255,0.05)')
        .attr('stroke', 'rgba(0,217,255,0.2)').attr('stroke-width', 1);
      g.append('text').attr('x', 460).attr('y', by + 26)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 12).attr('font-family', 'monospace')
        .text('\u2202L/\u2202b  =  \u2202L/\u2202a \u00b7 \u2202a/\u2202z \u00b7 \u2202z/\u2202b  =  (\u22120.378)(0.235)(1)  =  \u22120.0888');
      g.append('text').attr('x', 460).attr('y', by + 52)
        .attr('text-anchor', 'middle').attr('fill', '#4ade80')
        .attr('font-size', 12).attr('font-family', 'monospace')
        .text('b  \u2190  0.100 \u2212 0.1\u00d7(\u22120.0888)  =  0.1089  \u2713');
    });
  }

  // ── SLIDE: Weights & Biases — Layer as Matrix Multiplication ─────────────
  function initBackpropTensor(el) {
    el.innerHTML = '';
    const W = 1100, H = 510;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%').style('overflow', 'visible');

    const timers = [];
    el._timers = timers;
    const later = (ms, fn) => timers.push(setTimeout(fn, ms));

    addArrowDef(svg, 'bpt-fwd', C.primary);

    // ── layout ──────────────────────────────────────────────────────────────
    const CW = 38, CH = 28;         // cell dimensions
    const NI = 3, NO = 4;           // 3 inputs, 4 output neurons

    // Forward pass positions
    const WX = 55,  WY = 82;
    const AX = WX + NI*CW + 22,    AY = WY + (NO*CH - NI*CH)/2;
    const BX = AX + CW  + 22,      BY = WY;
    const ZX = BX + CW  + 22,      ZY = WY;
    const OX = ZX + CW  + 68,      OY = WY;
    const midY = WY + NO*CH / 2;   // vertical centre of expression

    // Backward pass positions
    const BPY = 248;
    const DWX = 55,    DWY = BPY;
    const DBX = DWX + NI*CW + 30,  DBY = BPY;
    const DPX = DBX + CW    + 30,  DPY = BPY;   // "propagates back" Wᵀδ

    // ── helpers ─────────────────────────────────────────────────────────────
    function fg(delay, build) {
      const g = svg.append('g').attr('opacity', 0);
      build(g);
      g.transition().delay(delay).duration(380).attr('opacity', 1);
      return g;
    }

    function cell(g, cx, cy, fill, stroke, txt, fs) {
      g.append('rect').attr('x', cx).attr('y', cy)
        .attr('width', CW - 3).attr('height', CH - 3).attr('rx', 3)
        .attr('fill', fill).attr('stroke', stroke).attr('stroke-width', 1.2);
      g.append('text').attr('x', cx + (CW-3)/2).attr('y', cy + (CH-3)/2 + 4)
        .attr('text-anchor', 'middle').attr('fill', 'rgba(248,249,250,0.92)')
        .attr('font-size', fs || 9.5).attr('font-family', 'monospace')
        .text(txt);
    }

    function grid(g, x, y, cols, rows, fill, stroke, vals, fs) {
      for (let r = 0; r < rows; r++)
        for (let c = 0; c < cols; c++)
          cell(g, x + c*CW, y + r*CH, fill, stroke, vals[r][c], fs);
    }

    function bracket(g, x, y, h, color, flip) {
      const d = flip ? -1 : 1;
      g.append('path')
        .attr('d', `M${x} ${y} L${x-d*6} ${y} L${x-d*6} ${y+h} L${x} ${y+h}`)
        .attr('fill', 'none').attr('stroke', color).attr('stroke-width', 1.5);
    }

    function matLabel(g, x, y, rows, cols, label, color) {
      // outer bracket pair
      bracket(g, x,           y, rows*CH, color, false);
      bracket(g, x + cols*CW, y, rows*CH, color, true);
      // dim label below
      g.append('text').attr('x', x + cols*CW/2).attr('y', y + rows*CH + 15)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8.5).attr('font-family', 'sans-serif').text(label);
    }

    function op(g, x, y, txt) {
      g.append('text').attr('x', x).attr('y', y)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('fill', C.muted).attr('font-size', 20).attr('font-family', 'sans-serif')
        .text(txt);
    }

    // ── phase label ─────────────────────────────────────────────────────────
    const phLbl = svg.append('text').attr('x', 480).attr('y', 30)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 16).attr('font-family', 'sans-serif').attr('font-style', 'italic')
      .text('one layer = matrix multiply + bias vector');

    // ══════ FORWARD PASS ════════════════════════════════════════════════════

    // W matrix
    const wV = [
      ['w\u2081\u2081','w\u2081\u2082','w\u2081\u2083'],
      ['w\u2082\u2081','w\u2082\u2082','w\u2082\u2083'],
      ['w\u2083\u2081','w\u2083\u2082','w\u2083\u2083'],
      ['w\u2084\u2081','w\u2084\u2082','w\u2084\u2083'],
    ];
    fg(0, g => {
      grid(g, WX, WY, NI, NO, 'rgba(157,78,221,0.18)', C.secondary, wV);
      matLabel(g, WX, WY, NO, NI, 'W  \u2208  \u211d^(n\u2092\u1d58\u1d57 \u00d7 n\u1d35\u2099)', C.secondary);
      g.append('text').attr('x', WX + NI*CW/2).attr('y', WY - 10)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 12).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('W');
    });

    // a input vector
    const aV = [['a\u2081'],['a\u2082'],['a\u2083']];
    fg(450, g => {
      op(g, AX - 12, midY, '\u00d7');
      grid(g, AX, AY, 1, NI, 'rgba(0,217,255,0.12)', C.primary, aV);
      matLabel(g, AX, AY, NI, 1, 'a  \u2208  \u211d^n\u1d35\u2099', C.primary);
      g.append('text').attr('x', AX + CW/2).attr('y', AY - 10)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 12).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('a');
    });

    // b bias vector
    const bV = [['b\u2081'],['b\u2082'],['b\u2083'],['b\u2084']];
    fg(900, g => {
      op(g, BX - 12, midY, '+');
      grid(g, BX, BY, 1, NO, 'rgba(255,0,110,0.10)', C.accent, bV);
      matLabel(g, BX, BY, NO, 1, 'b  \u2208  \u211d^n\u2092\u1d58\u1d57', C.accent);
      g.append('text').attr('x', BX + CW/2).attr('y', BY - 10)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 12).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('b');
    });

    // z = Wa+b result
    const zV = [['z\u2081'],['z\u2082'],['z\u2083'],['z\u2084']];
    fg(1350, g => {
      op(g, ZX - 12, midY, '=');
      grid(g, ZX, ZY, 1, NO, 'rgba(157,78,221,0.14)', C.secondary, zV);
      matLabel(g, ZX, ZY, NO, 1, 'z  (pre-act)', C.secondary);
      g.append('text').attr('x', ZX + CW/2).attr('y', ZY - 10)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 12).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('z');
    });

    // \u03c3(z) \u2192 a' output
    const oV = [['a\u2081\u2032'],['a\u2082\u2032'],['a\u2083\u2032'],['a\u2084\u2032']];
    fg(1800, g => {
      g.append('line').attr('x1', ZX + CW + 2).attr('y1', midY)
        .attr('x2', OX - 8).attr('y2', midY)
        .attr('stroke', C.primary).attr('stroke-width', 1.5)
        .attr('marker-end', 'url(#bpt-fwd)');
      g.append('text').attr('x', ZX + CW + (OX - ZX - CW)/2 + 2).attr('y', midY - 8)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 12).attr('font-family', 'serif').text('\u03c3( )');
      grid(g, OX, OY, 1, NO, 'rgba(74,222,128,0.12)', '#4ade80', oV);
      matLabel(g, OX, OY, NO, 1, "a'  (act)", '#4ade80');
      g.append('text').attr('x', OX + CW/2).attr('y', OY - 10)
        .attr('text-anchor', 'middle').attr('fill', '#4ade80')
        .attr('font-size', 12).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text("a'");
    });

    // ── right-side formula panel ─────────────────────────────────────────────
    fg(2000, g => {
      const rx = 570, ry = 55, rw = 310, rh = 196;
      g.append('rect').attr('x', rx).attr('y', ry).attr('width', rw).attr('height', rh)
        .attr('rx', 6).attr('fill', 'rgba(0,0,0,0.45)')
        .attr('stroke', 'rgba(255,255,255,0.10)').attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,3');
      [
        [ry+18,  C.muted,    10, false, 'forward'],
        [ry+38,  C.secondary,14, true,  'z = Wa + b'],
        [ry+57,  '#4ade80',  13, true,  "a' = \u03c3(z)"],
        [ry+84,  C.muted,    10, false, 'backward  (\u03b4 = \u2202L/\u2202z  \u2014  error signal, not the bias)'],
        [ry+104, C.accent,   14, true,  '\u2202L/\u2202W = \u03b4 a\u1d40          (same shape as W)'],
        [ry+123, C.accent,   14, true,  '\u2202L/\u2202b = \u03b4                (\u2202z/\u2202b = 1, so chain = \u03b4)'],
        [ry+142, C.primary,  11, false, '\u2202L/\u2202a = W\u1d40\u03b4              (signal to prev. layer)'],
        [ry+163, C.muted,    10, false, 'how \u03b4 is computed for a hidden layer:'],
        [ry+181, '#4ade80',  12, true,  '\u03b4 = W\u1d40\u03b4\u207f\u1d49\u02e3\u1d57 \u2299 \u03c3\u2032(z)'],
      ].forEach(([y, color, size, bold, text]) =>
        g.append('text').attr('x', rx + 14).attr('y', y)
          .attr('fill', color).attr('font-size', size)
          .attr('font-family', size >= 13 ? 'monospace' : 'sans-serif')
          .attr('font-weight', bold ? 'bold' : 'normal')
          .text(text));
    });

    // ══════ BACKWARD PASS ════════════════════════════════════════════════════
    later(2600, () => {
      phLbl.transition().duration(300)
        .attr('fill', C.accent).attr('font-style', 'normal')
        .text('\u2190 \u2202L/\u2202W has the same shape as W — every weight gets its own gradient');
    });

    // \u2202L/\u2202W matrix (same shape as W)
    const dwV = [
      ['\u03b4\u2081a\u2081','\u03b4\u2081a\u2082','\u03b4\u2081a\u2083'],
      ['\u03b4\u2082a\u2081','\u03b4\u2082a\u2082','\u03b4\u2082a\u2083'],
      ['\u03b4\u2083a\u2081','\u03b4\u2083a\u2082','\u03b4\u2083a\u2083'],
      ['\u03b4\u2084a\u2081','\u03b4\u2084a\u2082','\u03b4\u2084a\u2083'],
    ];
    fg(2600, g => {
      grid(g, DWX, DWY, NI, NO, 'rgba(255,0,110,0.12)', C.accent, dwV, 8.5);
      matLabel(g, DWX, DWY, NO, NI, '\u2202L/\u2202W  =  \u03b4 a\u1d40', C.accent);
      g.append('text').attr('x', DWX + NI*CW/2).attr('y', DWY - 10)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 11).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('\u2202L/\u2202W');
      // "same shape" arrow from W to \u2202L/\u2202W
      g.append('text').attr('x', DWX + NI*CW/2).attr('y', DWY - 24)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8.5).attr('font-family', 'sans-serif')
        .text('same shape as W \u2191');
    });

    // \u2202L/\u2202b vector
    const dbV = [['\u03b4\u2081'],['\u03b4\u2082'],['\u03b4\u2083'],['\u03b4\u2084']];
    fg(3200, g => {
      grid(g, DBX, DBY, 1, NO, 'rgba(255,0,110,0.12)', C.accent, dbV);
      matLabel(g, DBX, DBY, NO, 1, '\u2202L/\u2202b  =  \u03b4', C.accent);
      g.append('text').attr('x', DBX + CW/2).attr('y', DBY - 10)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 11).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('\u2202L/\u2202b');
      g.append('text').attr('x', DBX + CW/2).attr('y', DBY - 24)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8.5).attr('font-family', 'sans-serif')
        .text('= \u03b4  (since \u2202z/\u2202b = 1)');
    });

    // W\u1d40\u03b4 — error signal propagation (same W, transposed)
    // Wᵀ (NI rows × NO cols) · δ (NO×1) = ∂L/∂a (NI×1)
    const CWS = 28, CHS = 24;    // smaller cells for this section
    const WTX = 272, WTY = BPY;
    const centOff = Math.round((NO*CH - NI*CHS) / 2);  // vert. centre Wᵀ & result
    const dOff    = Math.round((NO*CH - NO*CHS) / 2);  // vert. centre δ

    later(3800, () => {
      phLbl.transition().duration(300)
        .attr('fill', C.primary).attr('font-style', 'normal')
        .text('\u2190 W\u1d40\u03b4: same weights, transposed — carry the error signal to the previous layer');
    });

    // Wᵀ matrix
    const wtV = [
      ['w\u2081\u2081','w\u2082\u2081','w\u2083\u2081','w\u2084\u2081'],
      ['w\u2081\u2082','w\u2082\u2082','w\u2083\u2082','w\u2084\u2082'],
      ['w\u2081\u2083','w\u2082\u2083','w\u2083\u2083','w\u2084\u2083'],
    ];
    fg(3800, g => {
      const wx = WTX, wy = WTY + centOff;
      // highlight relationship: same cells as W but transposed
      g.append('rect').attr('x', wx - 3).attr('y', wy - 3)
        .attr('width', NO*CWS + 4).attr('height', NI*CHS + 4).attr('rx', 5)
        .attr('fill', 'none').attr('stroke', C.primary).attr('stroke-width', 1.6);
      for (let r = 0; r < NI; r++)
        for (let c = 0; c < NO; c++) {
          g.append('rect').attr('x', wx + c*CWS).attr('y', wy + r*CHS)
            .attr('width', CWS-2).attr('height', CHS-2).attr('rx', 3)
            .attr('fill', 'rgba(157,78,221,0.15)').attr('stroke', C.primary).attr('stroke-width', 1);
          g.append('text').attr('x', wx + c*CWS + (CWS-2)/2).attr('y', wy + r*CHS + (CHS-2)/2 + 4)
            .attr('text-anchor', 'middle').attr('fill', 'rgba(248,249,250,0.9)')
            .attr('font-size', 8.5).attr('font-family', 'monospace')
            .text(wtV[r][c]);
        }
      g.append('text').attr('x', wx + NO*CWS/2).attr('y', wy - 14)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 11).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('W\u1d40');
      g.append('text').attr('x', wx + NO*CWS/2).attr('y', wy + NI*CHS + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'sans-serif')
        .text('(n\u1d35\u2099 \u00d7 n\u2092\u1d58\u1d57) \u2014 same weights, flipped');
    });

    // δ vector
    const dvV = [['\u03b4\u2081'],['\u03b4\u2082'],['\u03b4\u2083'],['\u03b4\u2084']];
    fg(4300, g => {
      const dx = WTX + NO*CWS + 14, dy = WTY + dOff;
      g.append('text').attr('x', dx - 8).attr('y', WTY + NO*CH/2)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('fill', C.muted).attr('font-size', 18).attr('font-family', 'sans-serif').text('\u00d7');
      g.append('rect').attr('x', dx - 2).attr('y', dy - 2)
        .attr('width', CWS + 2).attr('height', NO*CHS + 2).attr('rx', 5)
        .attr('fill', 'none').attr('stroke', C.accent).attr('stroke-width', 1.6);
      for (let r = 0; r < NO; r++) {
        g.append('rect').attr('x', dx).attr('y', dy + r*CHS)
          .attr('width', CWS-2).attr('height', CHS-2).attr('rx', 3)
          .attr('fill', 'rgba(255,0,110,0.12)').attr('stroke', C.accent).attr('stroke-width', 1);
        g.append('text').attr('x', dx + (CWS-2)/2).attr('y', dy + r*CHS + (CHS-2)/2 + 4)
          .attr('text-anchor', 'middle').attr('fill', 'rgba(248,249,250,0.9)')
          .attr('font-size', 9).attr('font-family', 'monospace')
          .text(dvV[r][0]);
      }
      g.append('text').attr('x', dx + CWS/2).attr('y', dy - 14)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 11).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('\u03b4');
      g.append('text').attr('x', dx + CWS/2).attr('y', dy + NO*CHS + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'sans-serif')
        .text('error signal');
    });

    // = ∂L/∂a result
    const daV = [['\u2202a\u2081'],['\u2202a\u2082'],['\u2202a\u2083']];
    fg(4900, g => {
      const rx = WTX + NO*CWS + 14 + CWS + 18, ry = WTY + centOff;
      g.append('text').attr('x', rx - 10).attr('y', WTY + NO*CH/2)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('fill', C.muted).attr('font-size', 18).attr('font-family', 'sans-serif').text('=');
      g.append('rect').attr('x', rx - 2).attr('y', ry - 2)
        .attr('width', CWS + 2).attr('height', NI*CHS + 2).attr('rx', 5)
        .attr('fill', 'none').attr('stroke', C.primary).attr('stroke-width', 1.8);
      for (let r = 0; r < NI; r++) {
        g.append('rect').attr('x', rx).attr('y', ry + r*CHS)
          .attr('width', CWS-2).attr('height', CHS-2).attr('rx', 3)
          .attr('fill', 'rgba(0,217,255,0.12)').attr('stroke', C.primary).attr('stroke-width', 1);
        g.append('text').attr('x', rx + (CWS-2)/2).attr('y', ry + r*CHS + (CHS-2)/2 + 4)
          .attr('text-anchor', 'middle').attr('fill', 'rgba(248,249,250,0.9)')
          .attr('font-size', 8.5).attr('font-family', 'monospace')
          .text(daV[r][0]);
      }
      g.append('text').attr('x', rx + CWS/2).attr('y', ry - 14)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 11).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('\u2202L/\u2202a');
      g.append('text').attr('x', rx + CWS/2).attr('y', ry + NI*CHS + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'sans-serif')
        .text('\u2192 prev. layer');
    });

    // ⊙ σ′(z): element-wise activation derivative turns incoming signal into δ
    // Positions: continue rightward from ∂L/∂a result (which ends at WTX+NO*CWS+14+CWS+18+CWS)
    const daEndX = WTX + NO*CWS + 14 + CWS + 18 + CWS;   // 472
    const spStartX = daEndX + 18;                          // 490

    later(5200, () => {
      phLbl.transition().duration(300)
        .attr('fill', '#4ade80').attr('font-style', 'normal')
        .text('\u2299 \u03c3\u2032(z): element-wise activation slope turns the incoming signal into \u03b4');
    });

    // σ′(z) vector (NI rows × 1 col, same shape as ∂L/∂a)
    const spV = [["\u03c3'\u2081"],["\u03c3'\u2082"],["\u03c3'\u2083"]];
    fg(5200, g => {
      const sx = spStartX, sy = WTY + centOff;
      g.append('text').attr('x', sx - 10).attr('y', WTY + NO*CH / 2)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('fill', C.muted).attr('font-size', 16).attr('font-family', 'sans-serif')
        .text('\u2299');
      g.append('rect').attr('x', sx - 2).attr('y', sy - 2)
        .attr('width', CWS + 2).attr('height', NI*CHS + 2).attr('rx', 5)
        .attr('fill', 'none').attr('stroke', C.secondary).attr('stroke-width', 1.6);
      for (let r = 0; r < NI; r++) {
        g.append('rect').attr('x', sx).attr('y', sy + r*CHS)
          .attr('width', CWS-2).attr('height', CHS-2).attr('rx', 3)
          .attr('fill', 'rgba(157,78,221,0.14)').attr('stroke', C.secondary).attr('stroke-width', 1);
        g.append('text').attr('x', sx + (CWS-2)/2).attr('y', sy + r*CHS + (CHS-2)/2 + 4)
          .attr('text-anchor', 'middle').attr('fill', 'rgba(248,249,250,0.9)')
          .attr('font-size', 9).attr('font-family', 'monospace')
          .text(spV[r][0]);
      }
      g.append('text').attr('x', sx + CWS/2).attr('y', sy - 14)
        .attr('text-anchor', 'middle').attr('fill', C.secondary)
        .attr('font-size', 10).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text("\u03c3\u2032(z)");
      g.append('text').attr('x', sx + CWS/2).attr('y', sy + NI*CHS + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'sans-serif')
        .text('act. slope');
    });

    // = δ result for the receiving (previous) layer
    const dpV2 = [['\u03b4\u2081'],['\u03b4\u2082'],['\u03b4\u2083']];
    fg(5600, g => {
      const ex = spStartX + CWS + 18, ey = WTY + centOff;
      g.append('text').attr('x', ex - 10).attr('y', WTY + NO*CH / 2)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .attr('fill', C.muted).attr('font-size', 18).attr('font-family', 'sans-serif')
        .text('=');
      g.append('rect').attr('x', ex - 2).attr('y', ey - 2)
        .attr('width', CWS + 2).attr('height', NI*CHS + 2).attr('rx', 5)
        .attr('fill', 'none').attr('stroke', C.accent).attr('stroke-width', 2);
      for (let r = 0; r < NI; r++) {
        g.append('rect').attr('x', ex).attr('y', ey + r*CHS)
          .attr('width', CWS-2).attr('height', CHS-2).attr('rx', 3)
          .attr('fill', 'rgba(255,0,110,0.16)').attr('stroke', C.accent).attr('stroke-width', 1.2);
        g.append('text').attr('x', ex + (CWS-2)/2).attr('y', ey + r*CHS + (CHS-2)/2 + 4)
          .attr('text-anchor', 'middle').attr('fill', 'rgba(248,249,250,0.9)')
          .attr('font-size', 9).attr('font-family', 'monospace')
          .text(dpV2[r][0]);
      }
      g.append('text').attr('x', ex + CWS/2).attr('y', ey - 14)
        .attr('text-anchor', 'middle').attr('fill', C.accent)
        .attr('font-size', 11).attr('font-weight', 'bold').attr('font-family', 'monospace')
        .text('\u03b4  (prev. layer)');
      g.append('text').attr('x', ex + CWS/2).attr('y', ey + NI*CHS + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8).attr('font-family', 'sans-serif')
        .text('chain continues \u2190');
    });

    // update banner
    later(6400, () => {
      phLbl.transition().duration(300).attr('fill', '#4ade80')
        .text('one matrix op updates every weight and bias in the layer simultaneously');
    });

    fg(6400, g => {
      const by = BPY + NO*CH + 26;
      g.append('rect').attr('x', 55).attr('y', by).attr('width', 600).attr('height', 54)
        .attr('rx', 6).attr('fill', 'rgba(0,217,255,0.05)')
        .attr('stroke', 'rgba(0,217,255,0.20)').attr('stroke-width', 1);
      g.append('text').attr('x', 355).attr('y', by + 22)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 12.5).attr('font-family', 'monospace').attr('font-weight', 'bold')
        .text('W  \u2190  W \u2212 \u03b1 \u22c5 \u03b4 a\u1d40');
      g.append('text').attr('x', 355).attr('y', by + 42)
        .attr('text-anchor', 'middle').attr('fill', '#4ade80')
        .attr('font-size', 12.5).attr('font-family', 'monospace').attr('font-weight', 'bold')
        .text('b  \u2190  b \u2212 \u03b1 \u22c5 \u03b4');
    });
  }
  // ── SLIDE: Mini-Batches & Epochs ─────────────────────────────────────────
  function initBatchEpochs(el) {
    el.innerHTML = '';
    const W = 1100, H = 510;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%').style('overflow', 'visible');

    const timers = [];
    el._timers = timers;
    const later = (ms, fn) => timers.push(setTimeout(fn, ms));

    function fg(delay, build) {
      const g = svg.append('g').attr('opacity', 0);
      build(g);
      g.transition().delay(delay).duration(380).attr('opacity', 1);
      return g;
    }

    // ── constants ────────────────────────────────────────────────────────────
    const N = 24, B = 4, NB = N / B;       // 24 samples, batch size 4 = 6 steps
    const SR = 8;                            // sample dot radius
    const SY = 90, SX0 = 55, SX1 = 1045;
    const SP = (SX1 - SX0) / N;             // px per sample
    const BW = SP * B;                       // px per batch

    const COL_Y = 155;
    const C1X = 55, C2X = 398, C3X = 745;
    const CW = 308;

    // ── phase label ──────────────────────────────────────────────────────────
    const phLbl = svg.append('text').attr('x', 550).attr('y', 30)
      .attr('text-anchor', 'middle').attr('fill', C.muted)
      .attr('font-size', 16).attr('font-family', 'sans-serif').attr('font-style', 'italic')
      .text('each weight update uses B samples \u2014 not one, not all');

    // Column separators (static)
    svg.append('g').call(g =>
      [C2X - 16, C3X - 16].forEach(x =>
        g.append('line').attr('x1', x).attr('y1', COL_Y - 4).attr('x2', x).attr('y2', 462)
          .attr('stroke', 'rgba(255,255,255,0.07)').attr('stroke-width', 1)));

    // ── Dataset strip ─────────────────────────────────────────────────────────
    const sampleColors = [
      'rgba(0,217,255,0.62)', 'rgba(157,78,221,0.62)',
      'rgba(74,222,128,0.62)', 'rgba(255,0,110,0.62)',
    ];

    fg(0, g => {
      g.append('rect').attr('x', SX0 - 14).attr('y', SY - SR - 5)
        .attr('width', SX1 - SX0 + 28).attr('height', (SR + 5) * 2)
        .attr('rx', 4).attr('fill', 'rgba(255,255,255,0.02)')
        .attr('stroke', 'rgba(255,255,255,0.05)').attr('stroke-width', 1);
      for (let i = 0; i < N; i++) {
        const cx = SX0 + (i + 0.5) * SP;
        g.append('circle').attr('cx', cx).attr('cy', SY).attr('r', SR)
          .attr('fill', sampleColors[i % 4])
          .attr('stroke', 'rgba(255,255,255,0.12)').attr('stroke-width', 0.8);
      }
      g.append('text').attr('x', 550).attr('y', SY - SR - 12)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9.5).attr('font-family', 'sans-serif')
        .text('full dataset  (N = ' + N + ' samples, re-shuffled every epoch)');
    });

    fg(500, g => {
      // Batch separators
      for (let b = 1; b < NB; b++)
        g.append('line').attr('x1', SX0 + b * BW).attr('y1', SY - SR - 6)
          .attr('x2', SX0 + b * BW).attr('y2', SY + SR + 6)
          .attr('stroke', 'rgba(255,255,255,0.14)').attr('stroke-width', 1)
          .attr('stroke-dasharray', '3,2');
      // Highlight batch 1
      g.append('rect').attr('x', SX0 - 5).attr('y', SY - SR - 5)
        .attr('width', BW + 10).attr('height', (SR + 5) * 2).attr('rx', 5)
        .attr('fill', 'rgba(0,217,255,0.06)').attr('stroke', C.primary).attr('stroke-width', 1.8);
      g.append('text').attr('x', SX0 + BW / 2).attr('y', SY - SR - 16)
        .attr('text-anchor', 'middle').attr('fill', C.primary)
        .attr('font-size', 9).attr('font-family', 'sans-serif').text('\u2193 step 1');
      // Step labels
      for (let b = 0; b < NB; b++)
        g.append('text').attr('x', SX0 + (b + 0.5) * BW).attr('y', SY + SR + 17)
          .attr('text-anchor', 'middle').attr('fill', b === 0 ? C.primary : C.muted)
          .attr('font-size', 8.5).attr('font-family', 'sans-serif')
          .text('step ' + (b + 1));
      g.append('text').attr('x', 550).attr('y', SY + SR + 32)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 9).attr('font-family', 'monospace')
        .text('1 epoch = ' + NB + ' steps  (N/B = ' + N + '/' + B + ')');
    });

    // \u2550\u2550 Col 1: Batch size trade-offs \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    fg(1100, g => {
      g.append('text').attr('x', C1X).attr('y', COL_Y + 16)
        .attr('fill', C.primary).attr('font-size', 12).attr('font-weight', 'bold')
        .attr('font-family', 'monospace').text('Batch size (B)');

      [
        { b: 'B = 1',       sub: 'stochastic GD',  l1: 'very noisy gradient direction',    l2: 'fast per-step, slow to converge', c: C.accent  },
        { b: 'B = 32\u2013256', sub: 'mini-batch \u2713', l1: 'smooth + GPU-parallelisable',    l2: 'standard for most networks',      c: C.primary },
        { b: 'B = N',       sub: 'full-batch GD',  l1: 'accurate gradient, no noise',      l2: 'huge memory, impractical at scale', c: C.muted   },
      ].forEach(({ b, sub, l1, l2, c }, i) => {
        const ry = COL_Y + 38 + i * 80;
        g.append('rect').attr('x', C1X).attr('y', ry)
          .attr('width', CW - 14).attr('height', 72).attr('rx', 5)
          .attr('fill', 'rgba(255,255,255,0.02)').attr('stroke', c + '55').attr('stroke-width', 1);
        g.append('text').attr('x', C1X + 10).attr('y', ry + 20)
          .attr('fill', c).attr('font-size', 11).attr('font-weight', 'bold')
          .attr('font-family', 'monospace').text(b + '  (' + sub + ')');
        [l1, l2].forEach((ln, li) =>
          g.append('text').attr('x', C1X + 10).attr('y', ry + 38 + li * 17)
            .attr('fill', C.muted).attr('font-size', 9).attr('font-family', 'sans-serif').text(ln));
      });
    });

    // \u2550\u2550 Col 2: Gradient averaging \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    fg(1700, g => {
      g.append('text').attr('x', C2X).attr('y', COL_Y + 16)
        .attr('fill', C.accent).attr('font-size', 12).attr('font-weight', 'bold')
        .attr('font-family', 'monospace').text('Gradient averaging');

      [
        { y: 40,  txt: 'each sample n computes its own gradient:',     c: C.muted,    sz: 9.5,  bold: false },
        { y: 60,  txt: '\u2202W\u207f = \u03b4\u207f a\u207f\u1d40   (per sample)',          c: C.secondary,sz: 12.5, bold: true  },
        { y: 88,  txt: 'then average across the batch:',                c: C.muted,    sz: 9.5,  bold: false },
        { y: 112, txt: '\u2202L/\u2202W = (1/B) \u03a3\u207f \u03b4\u207f a\u207f\u1d40',           c: C.accent,   sz: 14,   bold: true  },
        { y: 135, txt: '(1/B) keeps scale independent of batch size',   c: C.muted,    sz: 9,    bold: false },
        { y: 155, txt: 'update:',                                        c: C.muted,    sz: 9.5,  bold: false },
        { y: 174, txt: 'W \u2190 W \u2212 \u03b1 \u22c5 (1/B) \u03a3\u207f \u03b4\u207f a\u207f\u1d40',      c: '#4ade80',  sz: 13,   bold: true  },
        { y: 196, txt: 'b \u2190 b \u2212 \u03b1 \u22c5 (1/B) \u03a3\u207f \u03b4\u207f',              c: '#4ade80',  sz: 13,   bold: true  },
        { y: 225, txt: 'averaging reduces gradient noise \u2014',          c: C.muted,    sz: 9,    bold: false },
        { y: 239, txt: 'more reliable update direction per step',        c: C.muted,    sz: 9,    bold: false },
      ].forEach(({ y, txt, c, sz, bold }) =>
        g.append('text').attr('x', C2X).attr('y', COL_Y + y)
          .attr('fill', c).attr('font-size', sz).attr('font-family', sz >= 12 ? 'monospace' : 'sans-serif')
          .attr('font-weight', bold ? 'bold' : 'normal').text(txt));
    });

    // \u2550\u2550 Col 3: Epochs \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
    later(2400, () => {
      phLbl.transition().duration(300).attr('fill', C.secondary).attr('font-style', 'normal')
        .text('shuffle the dataset at the start of each epoch \u2014 order must not be learned');
    });

    fg(2400, g => {
      g.append('text').attr('x', C3X).attr('y', COL_Y + 16)
        .attr('fill', C.secondary).attr('font-size', 12).attr('font-weight', 'bold')
        .attr('font-family', 'monospace').text('Epochs');

      g.append('text').attr('x', C3X).attr('y', COL_Y + 36)
        .attr('fill', C.muted).attr('font-size', 9.5).attr('font-family', 'sans-serif')
        .text('1 epoch = full pass through all N samples = N/B steps');
      g.append('text').attr('x', C3X).attr('y', COL_Y + 53)
        .attr('fill', C.secondary).attr('font-size', 11.5).attr('font-weight', 'bold')
        .attr('font-family', 'monospace').text('repeat many epochs until loss converges');

      // Loss curve
      const lx = C3X, ly = COL_Y + 68, lw = CW - 8, lh = 110;
      g.append('rect').attr('x', lx).attr('y', ly).attr('width', lw).attr('height', lh)
        .attr('rx', 4).attr('fill', 'rgba(0,0,0,0.28)')
        .attr('stroke', 'rgba(255,255,255,0.08)').attr('stroke-width', 1);
      g.append('text').attr('x', lx + lw/2).attr('y', ly + lh + 14)
        .attr('text-anchor', 'middle').attr('fill', C.muted)
        .attr('font-size', 8.5).attr('font-family', 'sans-serif').text('epochs');
      g.append('text').attr('x', lx + 5).attr('y', ly + 11)
        .attr('fill', C.muted).attr('font-size', 8).attr('font-family', 'sans-serif').text('loss');

      const pts = (fn) => Array.from({length: 10}, (_, i) => {
        const t = i / 9;
        const x = lx + 12 + (lw - 24) * t;
        const y = ly + lh - 10 - (lh - 22) * Math.max(0, Math.min(1, fn(t)));
        return x + ',' + y;
      }).join(' ');

      g.append('polyline').attr('points', pts(t => 1 - Math.exp(-3.2 * t)))
        .attr('fill', 'none').attr('stroke', C.primary).attr('stroke-width', 2);
      g.append('polyline').attr('points', pts(t => {
        const base = (1 - Math.exp(-2.6 * t)) * 0.88;
        return base - (t > 0.62 ? (t - 0.62) * 0.38 : 0);
      })).attr('fill', 'none').attr('stroke', C.accent).attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '5,3');

      // Legend
      [[lx+10, C.primary, 2, null,   'train loss'],
       [lx+72, C.accent,  1.5,'5,3', 'val loss']].forEach(([lx2, col, sw, da, label]) => {
        const ln = g.append('line').attr('x1', lx2).attr('y1', ly+lh-13)
          .attr('x2', lx2+16).attr('y2', ly+lh-13)
          .attr('stroke', col).attr('stroke-width', sw);
        if (da) ln.attr('stroke-dasharray', da);
        g.append('text').attr('x', lx2+19).attr('y', ly+lh-9)
          .attr('fill', col).attr('font-size', 7.5).attr('font-family', 'sans-serif').text(label);
      });

      // Notes
      [
        [210, C.muted,  '10 \u2013 100 epochs typical for deep networks'],
        [228, C.muted,  'shuffle dataset at start of each epoch'],
        [246, C.accent, 'stop when val loss stops improving'],
        [264, C.muted,  '(early stopping \u2014 prevents overfitting)'],
      ].forEach(([dy, col, txt]) =>
        g.append('text').attr('x', C3X).attr('y', COL_Y + dy)
          .attr('fill', col).attr('font-size', 9.5).attr('font-family', 'sans-serif').text(txt));
    });

    later(3400, () => {
      phLbl.transition().duration(300).attr('fill', '#4ade80')
        .text('mini-batch + gradient averaging + epochs = stable, efficient training');
    });
  }
  // ── SLIDE: CNN 3D Architecture ──────────────────────────────────────────
  function initCNNArchitecture(el) {
    const W = 1100, H = 505;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width','100%').style('height','100%');

    // ── Oblique projection ────────────────────────────────────────────────────
    const OA = Math.PI / 6;
    const OS2 = 0.48;
    const oX = d => d * Math.cos(OA) * OS2;
    const oY = d => -d * Math.sin(OA) * OS2;

    // ── Feature maps from trained CNN (cnn-weights.js) ────────────────────────
    const _cnnS = window.CNN_SAMPLE || null;
    const _blank16 = () => Array.from({length:16}, () => new Array(49).fill(0));
    const FM = _cnnS ? {
      c1: _cnnS.fm_c1, p1: _cnnS.fm_p1,
      c2: _cnnS.fm_c2, p2: _cnnS.fm_p2,
    } : { c1:_blank16(), p1:_blank16(), c2:_blank16(), p2:_blank16() };
    const _cnnPixels = (_cnnS && _cnnS.pixels) || PIXEL_DATA;

    // ── Drawing helpers ───────────────────────────────────────────────────────

    // Simple solid cuboid — used only for Flatten→FC
    function drawCuboid(g, fx, fy, fw, fh, depth, col) {
      const ox=oX(depth),oy=oY(depth);
      g.append('polygon').attr('points',`${fx},${fy} ${fx+fw},${fy} ${fx+fw+ox},${fy+oy} ${fx+ox},${fy+oy}`)
        .attr('fill',col.t).attr('stroke','rgba(0,0,0,0.25)').attr('stroke-width',0.6);
      g.append('polygon').attr('points',`${fx+fw},${fy} ${fx+fw+ox},${fy+oy} ${fx+fw+ox},${fy+oy+fh} ${fx+fw},${fy+fh}`)
        .attr('fill',col.s).attr('stroke','rgba(0,0,0,0.25)').attr('stroke-width',0.6);
      g.append('rect').attr('x',fx).attr('y',fy).attr('width',fw).attr('height',fh)
        .attr('fill',col.f).attr('stroke','rgba(255,255,255,0.45)').attr('stroke-width',0.7);
    }

    // 16 thin slice panels, back→front; face resolution inferred from data length
    function drawSlicedLayer(g, fx, fy, fw, fh, col, fmaps) {
      const N=fmaps.length;
      const FH=Math.round(Math.sqrt(fmaps[0].length)), FW=FH;
      const SU=10, ST=3;                  // slice unit, slab thickness (depth units)
      const sox=oX(SU), soy=oY(SU);      // screen offset per slice
      const stx=oX(ST), sty=oY(ST);      // slab face depth in screen coords
      const cW=fw/FW, cH=fh/FH;

      const sliceGs = new Array(N);

      for(let i=N-1; i>=0; i--) {
        const bx=fx+i*sox, by=fy+i*soy;
        const sg = g.append('g');
        sliceGs[i] = sg;

        // Top face (thin sliver between this slice and the background)
        sg.append('polygon')
          .attr('points',`${bx},${by} ${bx+fw},${by} ${bx+fw+stx},${by+sty} ${bx+stx},${by+sty}`)
          .attr('fill',col.t).attr('stroke','rgba(0,0,0,0.45)').attr('stroke-width',0.5);
        // Right face (thin sliver)
        sg.append('polygon')
          .attr('points',`${bx+fw},${by} ${bx+fw+stx},${by+sty} ${bx+fw+stx},${by+sty+fh} ${bx+fw},${by+fh}`)
          .attr('fill',col.s).attr('stroke','rgba(0,0,0,0.45)').attr('stroke-width',0.5);

        // Front face: 7×7 feature map (cells tile exactly; +0.5 overlap avoids sub-pixel gaps)
        const fm=fmaps[i];
        for(let r=0;r<FH;r++) for(let c=0;c<FW;c++) {
          const b=Math.round(fm[r*FW+c]*255);
          sg.append('rect')
            .attr('x',bx+c*cW).attr('y',by+r*cH)
            .attr('width',cW+0.5).attr('height',cH+0.5)
            .attr('fill',`rgb(${b},${b},${b})`);
        }
        // Outline
        sg.append('rect').attr('x',bx).attr('y',by).attr('width',fw).attr('height',fh)
          .attr('fill','none').attr('stroke','rgba(255,255,255,0.42)').attr('stroke-width',0.7);
      }

      // Hover highlight outline — sits above all slice groups
      const hoverOutline = g.append('rect')
        .attr('x', fx).attr('y', fy).attr('width', fw).attr('height', fh)
        .attr('fill', 'none').attr('stroke', 'rgba(255,255,255,0)').attr('stroke-width', 2)
        .attr('pointer-events', 'none');

      // Feature-map popup — scaled-up view of the hovered slice
      const PCS = FH <= 7 ? 11 : FH <= 14 ? 7 : 4;  // px per cell
      const PW = FW * PCS, PH = FH * PCS, PPAD = 5;
      const popupG = g.append('g').attr('pointer-events', 'none').attr('opacity', 0);
      popupG.append('rect')
        .attr('width', PW + PPAD*2).attr('height', PH + PPAD*2 + 14)
        .attr('rx', 3)
        .attr('fill', 'rgba(8,8,18,0.93)')
        .attr('stroke', 'rgba(255,255,255,0.55)').attr('stroke-width', 1);
      const popupPixels = [];
      for(let r=0; r<FH; r++) for(let c=0; c<FW; c++) {
        popupPixels.push(
          popupG.append('rect')
            .attr('x', PPAD + c*PCS).attr('y', PPAD + r*PCS)
            .attr('width', PCS).attr('height', PCS)
            .attr('fill', 'rgb(0,0,0)')
        );
      }
      const popupLabel = popupG.append('text')
        .attr('x', PPAD + PW/2).attr('y', PPAD + PH + 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', 8.5).attr('font-family', 'monospace').attr('font-weight', 'bold')
        .attr('fill', 'rgba(255,255,255,0.8)');

      // Transparent hit area covering the full oblique extent of the stack
      g.append('rect')
        .attr('x', fx)
        .attr('y', fy + (N-1)*soy)          // top of rearmost slice (soy < 0 → higher)
        .attr('width', fw + (N-1)*sox + stx)
        .attr('height', fh - (N-1)*soy)     // spans from back-top to front-bottom
        .attr('fill', 'transparent')
        .on('mousemove', function(event) {
          const [mx] = d3.pointer(event, g.node());
          const idx = Math.max(0, Math.min(N-1, Math.round((mx - fx) / sox)));
          const bx = fx + idx*sox, by = fy + idx*soy;

          // Update outline
          hoverOutline
            .attr('x', bx).attr('y', by)
            .attr('stroke', 'rgba(255,255,255,0.88)');

          // Fill popup pixels from this slice's feature map
          const fm = fmaps[idx];
          popupPixels.forEach((p, pi) => {
            const b = Math.round(fm[pi] * 255);
            p.attr('fill', `rgb(${b},${b},${b})`);
          });
          popupLabel.text(`ch ${idx}`);

          // Position popup centered above the hovered slice front face
          const popW = PW + PPAD*2, popH = PH + PPAD*2 + 14;
          const px = Math.max(0, Math.min(1100 - popW, bx + fw/2 - popW/2));
          const py = by - popH - 8;
          popupG.attr('transform', `translate(${px},${py})`).attr('opacity', 1);
        })
        .on('mouseleave', function() {
          hoverOutline.attr('stroke', 'rgba(255,255,255,0)');
          popupG.attr('opacity', 0);
        });
    }

    // MNIST input: thin card with pixel grid on front face
    function drawMNISTInput(g, fx, fy, fw, fh, depth, col) {
      const ox2=oX(depth),oy2=oY(depth),cellW=fw/28,cellH=fh/28,gap=0.55;
      g.append('polygon').attr('points',`${fx},${fy} ${fx+fw},${fy} ${fx+fw+ox2},${fy+oy2} ${fx+ox2},${fy+oy2}`)
        .attr('fill',col.t).attr('stroke','rgba(0,0,0,0.15)').attr('stroke-width',0.5);
      g.append('polygon').attr('points',`${fx+fw},${fy} ${fx+fw+ox2},${fy+oy2} ${fx+fw+ox2},${fy+oy2+fh} ${fx+fw},${fy+fh}`)
        .attr('fill',col.s).attr('stroke','rgba(0,0,0,0.15)').attr('stroke-width',0.5);
      for(let r=0;r<28;r++) for(let c=0;c<28;c++) {
        const b=Math.round(_cnnPixels[r*28+c]*255);
        g.append('rect').attr('x',fx+c*cellW+gap/2).attr('y',fy+r*cellH+gap/2)
          .attr('width',cellW-gap).attr('height',cellH-gap).attr('fill',`rgb(${b},${b},${b})`);
      }
      g.append('rect').attr('x',fx).attr('y',fy).attr('width',fw).attr('height',fh)
        .attr('fill','none').attr('stroke','rgba(255,255,255,0.65)').attr('stroke-width',1.2);
    }

    // ── Layer definitions ─────────────────────────────────────────────────────
    const CY=248, SS=4, SU=10, ND=16*SU;   // ND=160: total depth for 16-ch layers
    const LAYERS = [
      { fw:28*SS, fh:28*SS, depth:7,
        col:{f:'rgba(155,158,168,0.60)',t:'rgba(198,200,210,0.70)',s:'rgba(105,108,118,0.60)'},
        label:'Input', sub:['28\u00d728\u00d71'], lc:C.muted, isMNIST:true },
      { fw:28*SS, fh:28*SS, depth:ND,
        col:{f:'rgba(0,145,192,0.52)',t:'rgba(0,178,220,0.66)',s:'rgba(0,95,142,0.52)'},
        label:'Conv2D', sub:['28\u00d728\u00d716','k=5, pad, ReLU'], lc:C.primary, fmKey:'c1' },
      { fw:14*SS, fh:14*SS, depth:ND,
        col:{f:'rgba(126,50,206,0.52)',t:'rgba(155,76,240,0.66)',s:'rgba(80,26,150,0.52)'},
        label:'MaxPool', sub:['14\u00d714\u00d716','size 2'], lc:C.secondary, fmKey:'p1' },
      { fw:14*SS, fh:14*SS, depth:ND,
        col:{f:'rgba(0,145,192,0.52)',t:'rgba(0,178,220,0.66)',s:'rgba(0,95,142,0.52)'},
        label:'Conv2D', sub:['14\u00d714\u00d716','k=3, pad, ReLU'], lc:C.primary, fmKey:'c2' },
      { fw:7*SS, fh:7*SS, depth:ND,
        col:{f:'rgba(126,50,206,0.52)',t:'rgba(155,76,240,0.66)',s:'rgba(80,26,150,0.52)'},
        label:'MaxPool', sub:['7\u00d77\u00d716','size 2'], lc:C.secondary, fmKey:'p2' },
      { fw:10, fh:88, depth:10,
        col:{f:'rgba(36,155,65,0.52)',t:'rgba(55,188,84,0.66)',s:'rgba(20,105,40,0.52)'},
        label:'Flatten\u2192FC', sub:['784\u2192128','ReLU'], lc:'#4ade80' },
    ];

    const GAP=28;
    const xs=[];
    let cx2=28;
    for(const l of LAYERS){xs.push(cx2); cx2+=l.fw+oX(l.depth)+GAP;}
    const outCX=cx2-GAP+55;

    const pred = _cnnS ? _cnnS.predicted : (_sample.predicted??MNIST_LABEL);
    const phLbl=svg.append('text').attr('x',W/2).attr('y',H-8).attr('text-anchor','middle')
      .attr('fill','rgba(0,0,0,0)').attr('font-size',11).attr('font-family','monospace');

    // ── Animate layers left → right ───────────────────────────────────────────
    LAYERS.forEach((l,li) => {
      const x=xs[li], y=CY-l.fh/2, delay=120+li*340;
      const g=svg.append('g').attr('opacity',0);
      g.transition().delay(delay).duration(320).attr('opacity',1);

      if(l.isMNIST)       drawMNISTInput(g, x, y, l.fw, l.fh, l.depth, l.col);
      else if(l.fmKey)    drawSlicedLayer(g, x, y, l.fw, l.fh, l.col, FM[l.fmKey]);
      else                drawCuboid(g, x, y, l.fw, l.fh, l.depth, l.col);

      const lx=x+(l.fw+oX(l.depth))/2, ly=CY+l.fh/2+16;
      g.append('text').attr('x',lx).attr('y',ly).attr('text-anchor','middle')
        .attr('fill',l.lc).attr('font-size',10.5).attr('font-weight','bold')
        .attr('font-family','monospace').text(l.label);
      l.sub.forEach((line,si) =>
        g.append('text').attr('x',lx).attr('y',ly+13+si*11).attr('text-anchor','middle')
          .attr('fill',C.muted).attr('font-size',8.5).attr('font-family','monospace').text(line));

      if(li<LAYERS.length-1) {
        const rx=x+l.fw+oX(l.depth);
        g.append('text').attr('x',rx+GAP/2-5).attr('y',CY+4)
          .attr('fill',C.muted).attr('font-size',13).text('\u2192');
      }
    });

    // Fan connections from FC to output
    const lastR=xs[5]+LAYERS[5].fw+oX(LAYERS[5].depth);
    const arrowG=svg.append('g').attr('opacity',0);
    arrowG.transition().delay(120+5*340).duration(320).attr('opacity',1);
    for(let i=0;i<5;i++) {
      const fromY=CY-40+(CY+40-CY+40)*i/4;
      arrowG.append('line').attr('x1',lastR+2).attr('y1',CY-40+(80)*i/4)
        .attr('x2',outCX-11).attr('y2',CY-44+i*22)
        .attr('stroke','rgba(74,222,128,0.18)').attr('stroke-width',0.8);
    }
    arrowG.append('text').attr('x',lastR+14).attr('y',CY+4)
      .attr('fill',C.muted).attr('font-size',11).text('\u2192');

    // ── Output: 10 circles + probability bars ─────────────────────────────────
    const outDelay=120+6*340, outSpacing=22, outTopY=CY-4.5*outSpacing;
    const sfmax = (_cnnS && _cnnS.probs) || (() => {
      const rawP=Array.from({length:10},(_,i)=>{
        if(i===pred) return 0.82;
        const d=Math.min(Math.abs(i-pred),10-Math.abs(i-pred));
        return Math.max(0.004,0.055/d);
      });
      return rawP.map(p=>p/rawP.reduce((a,b)=>a+b,0));
    })();

    const outG=svg.append('g').attr('opacity',0);
    outG.transition().delay(outDelay).duration(350).attr('opacity',1);
    outG.append('text').attr('x',outCX).attr('y',outTopY-16).attr('text-anchor','middle')
      .attr('fill',C.muted).attr('font-size',9.5).attr('font-family','monospace').text('Output');

    const bars=[];
    for(let d=0;d<10;d++) {
      const oy2=outTopY+d*outSpacing, isPred=d===pred;
      outG.append('circle').attr('cx',outCX).attr('cy',oy2).attr('r',9)
        .attr('fill',isPred?'rgba(74,222,128,0.22)':'rgba(255,255,255,0.05)')
        .attr('stroke',isPred?'#4ade80':'rgba(255,255,255,0.2)').attr('stroke-width',isPred?2:0.8);
      outG.append('text').attr('x',outCX).attr('y',oy2+3.5).attr('text-anchor','middle')
        .attr('font-size',8.5).attr('fill',isPred?'#4ade80':C.muted).attr('font-family','monospace').text(d);
      const bar=outG.append('rect').attr('x',outCX+12).attr('y',oy2-4.5)
        .attr('width',0).attr('height',9).attr('rx',2)
        .attr('fill',isPred?'rgba(74,222,128,0.55)':'rgba(255,255,255,0.1)');
      bars.push({bar,w:sfmax[d]*78,isPred,oy2});
    }
    setTimeout(()=>{
      bars.forEach(({bar,w})=>bar.transition().duration(480).attr('width',w));
      bars.forEach(({isPred,oy2,w},di)=>{
        if(!isPred&&w<6) return;
        outG.append('text').attr('x',outCX+12+w+4).attr('y',oy2+3.5)
          .attr('font-size',8).attr('fill',isPred?'#4ade80':C.muted)
          .attr('font-family','monospace').attr('opacity',0)
          .transition().delay(480).duration(200).attr('opacity',1)
          .text((sfmax[di]*100).toFixed(1)+'%');
      });
    }, outDelay+380);
    setTimeout(()=>{
      phLbl.text(`CNN \u2014 conv\u2192pool\u2192conv\u2192pool\u2192FC\u2192softmax  |  predicted digit: ${pred}`)
        .transition().duration(400).attr('fill','#4ade80');
    }, outDelay+1100);
  }

  // ── SLIDE: Convolution Detail — sliding, multiplying, summing + pooling ─
  function initConvDetails(el) {
    const W = 1100, H = 505;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width','100%').style('height','100%');

    // ── Data ──────────────────────────────────────────────────────────────
    const PROW = 10, PCOL = 10;
    const patch = [];
    for (let r = 0; r < 6; r++)
      for (let c = 0; c < 6; c++)
        patch.push(PIXEL_DATA[(PROW+r)*28+(PCOL+c)]);

    const KERN = [[-1,0,1],[-2,0,2],[-1,0,1]];
    const KDISP = [['-1','0','+1'],['-2','0','+2'],['-1','0','+1']];

    const rawOut = [];
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++) {
        let v = 0;
        for (let kr = 0; kr < 3; kr++)
          for (let kc = 0; kc < 3; kc++)
            v += patch[(r+kr)*6+(c+kc)] * KERN[kr][kc];
        rawOut.push(v);
      }
    let rmn = Infinity, rmx = -Infinity;
    for (const v of rawOut) { if(v<rmn)rmn=v; if(v>rmx)rmx=v; }
    const convOut = rawOut.map(v => rmx===rmn ? 0 : (v-rmn)/(rmx-rmn));

    const maxOut = [
      Math.max(convOut[0],convOut[1],convOut[4],convOut[5]),
      Math.max(convOut[2],convOut[3],convOut[6],convOut[7]),
      Math.max(convOut[8],convOut[9],convOut[12],convOut[13]),
      Math.max(convOut[10],convOut[11],convOut[14],convOut[15]),
    ];
    const avgOut = [
      (convOut[0]+convOut[1]+convOut[4]+convOut[5])/4,
      (convOut[2]+convOut[3]+convOut[6]+convOut[7])/4,
      (convOut[8]+convOut[9]+convOut[12]+convOut[13])/4,
      (convOut[10]+convOut[11]+convOut[14]+convOut[15])/4,
    ];

    // ── Layout ─────────────────────────────────────────────────────────────
    const CS = 22, KS = 22, OS = 28;
    const IX = 20, IY = 78;
    const KX = IX+6*CS+18, KY = IY+(6*CS-3*KS)/2;
    const OX = KX+3*KS+20, OY = IY+(6*CS-4*OS)/2;
    const CONV_W = OX+4*OS;   // ≈368

    const PDIV  = CONV_W+25;
    const MIS   = 18;          // pool block cell size
    const RS    = 32;          // result cell size  (2*RS ≈ 4*MIS=72)
    const PIX   = PDIV+20;    // pool diagram x
    const POOL1Y = 100;        // max pool top y
    const POOL2Y = 280;        // avg pool top y
    const RARROW = PIX+4*MIS+8;
    const RX    = RARROW+22;   // result x
    const DESCX = RX+2*RS+16;

    // Block colors: [top-left, top-right, bot-left, bot-right]
    const BC = ['rgba(0,217,255,','rgba(157,78,221,','rgba(255,165,0,','rgba(255,0,110,'];
    function blockOf(r, c) { return (r<2?0:2)+(c<2?0:1); }

    // ── Section headers & divider ───────────────────────────────────────────
    svg.append('line')
      .attr('x1',PDIV).attr('x2',PDIV).attr('y1',10).attr('y2',H-10)
      .attr('stroke','rgba(255,255,255,0.12)').attr('stroke-width',1)
      .attr('stroke-dasharray','4,3');

    svg.append('text').attr('x',(IX+CONV_W)/2).attr('y',30)
      .attr('text-anchor','middle').attr('fill',C.primary)
      .attr('font-size',13).attr('font-weight','bold').attr('font-family','monospace')
      .text('Convolution');
    svg.append('text').attr('x',PDIV+(W-PDIV)/2).attr('y',30)
      .attr('text-anchor','middle').attr('fill',C.secondary)
      .attr('font-size',13).attr('font-weight','bold').attr('font-family','monospace')
      .text('Pooling (downsampling)');

    // ── Input grid ─────────────────────────────────────────────────────────
    const inpG = svg.append('g').attr('opacity',0);
    inpG.transition().duration(400).attr('opacity',1);

    inpG.append('text').attr('x',IX+6*CS/2).attr('y',IY-8)
      .attr('text-anchor','middle').attr('fill',C.muted).attr('font-size',10)
      .attr('font-family','monospace').text('6\u00d76 input patch');

    for (let r = 0; r < 6; r++)
      for (let c = 0; c < 6; c++) {
        const v = patch[r*6+c], b = Math.round(v*255);
        inpG.append('rect')
          .attr('x',IX+c*CS).attr('y',IY+r*CS)
          .attr('width',CS-1).attr('height',CS-1)
          .attr('fill',`rgb(${b},${b},${b})`)
          .attr('stroke','rgba(255,255,255,0.12)').attr('stroke-width',0.5);
        if (v > 0.08)
          inpG.append('text')
            .attr('x',IX+c*CS+CS/2).attr('y',IY+r*CS+CS*0.66)
            .attr('text-anchor','middle').attr('font-size',7)
            .attr('fill',v>0.5?'rgba(0,0,0,0.8)':'rgba(255,255,255,0.55)')
            .attr('font-family','monospace').text(v.toFixed(2));
      }

    // ── Kernel ─────────────────────────────────────────────────────────────
    const kernG = svg.append('g').attr('opacity',0);
    kernG.transition().delay(350).duration(400).attr('opacity',1);

    kernG.append('text').attr('x',KX+3*KS/2).attr('y',KY-8)
      .attr('text-anchor','middle').attr('fill',C.muted).attr('font-size',10)
      .attr('font-family','monospace').text('kernel');

    KERN.forEach((row,kr) => row.forEach((val,kc) => {
      const ab = Math.abs(val)/2;
      const fc = val<0?`rgba(157,78,221,${0.25+ab*0.6})`:val>0?`rgba(0,217,255,${0.25+ab*0.6})`:'rgba(255,255,255,0.06)';
      kernG.append('rect')
        .attr('x',KX+kc*KS).attr('y',KY+kr*KS)
        .attr('width',KS-1).attr('height',KS-1).attr('fill',fc)
        .attr('stroke','rgba(255,255,255,0.2)').attr('stroke-width',0.5);
      kernG.append('text')
        .attr('x',KX+kc*KS+KS/2).attr('y',KY+kr*KS+KS*0.66)
        .attr('text-anchor','middle').attr('font-size',9)
        .attr('fill','rgba(255,255,255,0.9)').attr('font-family','monospace')
        .text(KDISP[kr][kc]);
    }));

    // ─ × and = symbols ──────────────────────────────────────────────────────
    const symsG = svg.append('g').attr('opacity',0);
    symsG.transition().delay(350).duration(400).attr('opacity',1);
    const SY = IY+6*CS/2+5;
    symsG.append('text').attr('x',KX-11).attr('y',SY)
      .attr('text-anchor','middle').attr('font-size',18).attr('fill',C.muted).text('\u00d7');
    symsG.append('text').attr('x',OX-12).attr('y',SY)
      .attr('text-anchor','middle').attr('font-size',18).attr('fill',C.muted).text('=');

    // ── Output grid ────────────────────────────────────────────────────────
    const outG = svg.append('g').attr('opacity',0);
    outG.transition().delay(350).duration(400).attr('opacity',1);

    outG.append('text').attr('x',OX+4*OS/2).attr('y',OY-8)
      .attr('text-anchor','middle').attr('fill',C.muted).attr('font-size',10)
      .attr('font-family','monospace').text('4\u00d74 output');

    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++)
        outG.append('rect')
          .attr('x',OX+c*OS).attr('y',OY+r*OS)
          .attr('width',OS-1).attr('height',OS-1)
          .attr('fill','rgba(255,255,255,0.03)')
          .attr('stroke','rgba(255,255,255,0.1)').attr('stroke-width',0.5);

    const outCells = [];
    for (let i = 0; i < 16; i++) {
      const r = Math.floor(i/4), c = i%4;
      outCells.push(svg.append('rect')
        .attr('x',OX+c*OS).attr('y',OY+r*OS)
        .attr('width',OS-1).attr('height',OS-1)
        .attr('fill','rgba(0,0,0,0)'));
    }

    const scanHi = svg.append('rect')
      .attr('width',3*CS-1).attr('height',3*CS-1)
      .attr('fill','rgba(0,217,255,0.10)').attr('stroke',C.primary)
      .attr('stroke-width',2).attr('opacity',0);
    const outHi = svg.append('rect')
      .attr('width',OS-1).attr('height',OS-1)
      .attr('fill','none').attr('stroke',C.accent)
      .attr('stroke-width',2).attr('opacity',0);

    // Info box
    const infoG = svg.append('g').attr('opacity',0);
    infoG.append('rect')
      .attr('x',IX).attr('y',IY+6*CS+14).attr('rx',4)
      .attr('width',CONV_W-IX).attr('height',52)
      .attr('fill','rgba(0,0,0,0.35)')
      .attr('stroke','rgba(255,255,255,0.07)').attr('stroke-width',0.5);
    const infoL1 = infoG.append('text')
      .attr('x',IX+8).attr('y',IY+6*CS+34)
      .attr('fill',C.text).attr('font-size',10.5).attr('font-family','monospace');
    const infoL2 = infoG.append('text')
      .attr('x',IX+8).attr('y',IY+6*CS+52)
      .attr('fill',C.muted).attr('font-size',9).attr('font-family','monospace');

    const phLbl = svg.append('text')
      .attr('x',W/2).attr('y',H-8).attr('text-anchor','middle')
      .attr('fill','rgba(0,0,0,0)').attr('font-size',11).attr('font-family','monospace');

    // ── Step-by-step convolution scan ──────────────────────────────────────
    let step = 0;
    function doStep() {
      if (step >= 16) {
        scanHi.transition().duration(200).attr('opacity',0);
        outHi.transition().duration(200).attr('opacity',0);
        infoG.transition().duration(300).attr('opacity',0);
        setTimeout(showPooling, 500);
        return;
      }
      const sr = Math.floor(step/4), sc2 = step%4;
      scanHi.attr('x',IX+sc2*CS).attr('y',IY+sr*CS).attr('opacity',1);
      outHi.attr('x',OX+sc2*OS).attr('y',OY+sr*OS).attr('opacity',1);

      let sum = 0; const ps = [];
      for (let kr = 0; kr < 3; kr++)
        for (let kc = 0; kc < 3; kc++) {
          const pv = patch[(sr+kr)*6+(sc2+kc)];
          sum += pv * KERN[kr][kc];
          ps.push((pv*KERN[kr][kc]).toFixed(2));
        }
      infoG.attr('opacity',1);
      infoL1.text(`\u03a3(patch \u00d7 kernel) = ${ps.slice(0,4).join(' + ')} + \u2026 = ${sum.toFixed(3)}`);
      infoL2.text(`output[${sr},${sc2}] (normalised) = ${convOut[step].toFixed(3)}`);
      const bv = Math.round(convOut[step]*255);
      outCells[step].attr('fill',`rgb(${bv},${bv},${bv})`);
      step++;
      setTimeout(doStep, 160);
    }
    setTimeout(doStep, 800);

    // ── Pooling section ────────────────────────────────────────────────────
    function showPooling() {
      // Shared pool labels
      [[POOL1Y, C.primary,   'Max Pooling (2\u00d72, stride 2)',
                             'take the maximum in each 2\u00d72 block \u2014 preserves strongest activation'],
       [POOL2Y, '#4ade80',   'Avg Pooling (2\u00d72, stride 2)',
                             'average each 2\u00d72 block \u2014 smoother, retains more spatial info'],
      ].forEach(([py, col, title, sub]) => {
        svg.append('text').attr('x',PIX).attr('y',py-14)
          .attr('text-anchor','start').attr('fill',col).attr('font-size',11.5)
          .attr('font-weight','bold').attr('font-family','monospace')
          .attr('opacity',0).transition().duration(400).attr('opacity',1).text(title);
        svg.append('text').attr('x',PIX).attr('y',py+4*MIS+14)
          .attr('text-anchor','start').attr('fill',C.muted).attr('font-size',9)
          .attr('font-family','monospace')
          .attr('opacity',0).transition().duration(400).attr('opacity',1).text(sub);
        svg.append('text').attr('x',RX+RS).attr('y',py-3)
          .attr('text-anchor','middle').attr('fill',C.muted).attr('font-size',9)
          .attr('font-family','monospace')
          .attr('opacity',0).transition().duration(400).attr('opacity',1).text('2\u00d72 result');
      });

      // Animate 4 blocks for both pool types
      setTimeout(() => {
        [0,1,2,3].forEach((bi, ii) => {
          setTimeout(() => {
            const col = BC[bi]+'0.85)';
            const br = bi<2?0:2, bcc = bi%2===0?0:2;
            const rr = bi<2?0:1, rc = bi%2;
            const bG = svg.append('g').attr('opacity',0);
            bG.transition().duration(220).attr('opacity',1);

            [POOL1Y, POOL2Y].forEach((poolY, pi) => {
              const resultArr = pi===0 ? maxOut : avgOut;
              // Draw 4 cells belonging to this block
              for (let r = 0; r < 4; r++)
                for (let c = 0; c < 4; c++) {
                  if (blockOf(r,c) !== bi) continue;
                  const v = convOut[r*4+c], bv = Math.round(v*255);
                  bG.append('rect')
                    .attr('x',PIX+c*MIS).attr('y',poolY+r*MIS)
                    .attr('width',MIS-1).attr('height',MIS-1)
                    .attr('fill',`rgb(${bv},${bv},${bv})`)
                    .attr('stroke',col).attr('stroke-width',1.5);
                }
              // Block outline
              bG.append('rect')
                .attr('x',PIX+bcc*MIS-1).attr('y',poolY+br*MIS-1)
                .attr('width',2*MIS+2).attr('height',2*MIS+2)
                .attr('fill','none').attr('stroke',col).attr('stroke-width',2);
              // Arrow
              bG.append('text')
                .attr('x',RARROW).attr('y',poolY+(br+1)*MIS+3)
                .attr('fill',col).attr('font-size',13).text('\u2192');
              // Result cell (vertically centred in 4*MIS block)
              const rvOff = Math.round((4*MIS-2*RS)/2);
              const rv = resultArr[bi], rb = Math.round(rv*255);
              bG.append('rect')
                .attr('x',RX+rc*RS).attr('y',poolY+rvOff+rr*RS)
                .attr('width',RS-1).attr('height',RS-1)
                .attr('fill',`rgb(${rb},${rb},${rb})`)
                .attr('stroke',col).attr('stroke-width',1.5);
              bG.append('text')
                .attr('x',RX+rc*RS+RS/2).attr('y',poolY+rvOff+rr*RS+RS*0.66)
                .attr('text-anchor','middle').attr('font-size',9)
                .attr('fill',rv>0.5?'rgba(0,0,0,0.8)':'rgba(255,255,255,0.7)')
                .attr('font-family','monospace').text(rv.toFixed(2));
            });
          }, ii*220);
        });

        // Descriptions after all blocks appear
        setTimeout(() => {
          [[POOL1Y, 'max: keeps the strongest\ndetected signal in the region'],
           [POOL2Y, 'avg: retains spatial average\nless aggressive downsampling'],
          ].forEach(([py, txt]) => {
            txt.split('\n').forEach((line, li) => {
              svg.append('text').attr('x',DESCX).attr('y',py+4*MIS/2-6+li*16)
                .attr('dominant-baseline','middle').attr('fill',C.muted)
                .attr('font-size',10.5).attr('font-family','monospace')
                .attr('opacity',0).transition().duration(300).attr('opacity',1).text(line);
            });
          });
          phLbl.text('pooling reduces spatial size \u2014 both types halve H\u00d7W, trading detail for efficiency')
            .transition().duration(400).attr('fill',C.muted);
        }, 4*220+250);
      }, 350);
    }
  }

  // ── SLIDE: Full Digit — Convolution & Max Pooling ────────────────────────
  function initConvFullDigit(el) {
    el.innerHTML = '';
    const W = 1100, H = 505;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width','100%').style('height','100%');

    addArrowDef(svg, 'arr-cfd', C.secondary);

    // ── Sobel-X kernel (same as conv-details) ─────────────────────────────────
    const KERN  = [[-1,0,1],[-2,0,2],[-1,0,1]];
    const KDISP = [['-1','0','+1'],['-2','0','+2'],['-1','0','+1']];

    // ── Compute full 28×28 convolution (same-pad) + ReLU ─────────────────────
    const convRaw = new Float32Array(784);
    for (let r = 0; r < 28; r++)
      for (let c = 0; c < 28; c++) {
        let v = 0;
        for (let kr = 0; kr < 3; kr++)
          for (let kc = 0; kc < 3; kc++) {
            const pr = r+kr-1, pc = c+kc-1;
            if (pr >= 0 && pr < 28 && pc >= 0 && pc < 28)
              v += PIXEL_DATA[pr*28+pc] * KERN[kr][kc];
          }
        convRaw[r*28+c] = Math.max(0, v);
      }
    const cmax = Math.max(...convRaw) || 1;
    const convOut = Array.from(convRaw, v => v / cmax);

    // ── 2×2 max pool stride 2 → 14×14 ────────────────────────────────────────
    const maxPool = [];
    for (let r = 0; r < 14; r++)
      for (let c = 0; c < 14; c++)
        maxPool.push(Math.max(
          convOut[(r*2)*28+c*2],   convOut[(r*2)*28+c*2+1],
          convOut[(r*2+1)*28+c*2], convOut[(r*2+1)*28+c*2+1]
        ));

    // ── Layout ────────────────────────────────────────────────────────────────
    const CS = 8;                     // input / conv cell px
    const GH = 28 * CS;               // 224 px
    const KS = 16;                    // kernel cell px
    const PS = 16;                    // pool cell px  (14*16 = 224 px)

    const IX = 15,  IY = Math.round((H - GH) / 2) - 4;   // ~136
    const KX = IX + 28*CS + 14;       // 253
    const KY = IY + Math.round((GH - 3*KS) / 2);          // centred
    const OX = KX + 3*KS + 18;       // 319
    const OY = IY;
    const PX = OX + 28*CS + 46;      // 589
    const PY = IY + Math.round((GH - 14*PS) / 2);         // same top as grids

    // ── Section labels ────────────────────────────────────────────────────────
    [
      [IX  + 14*CS, '28\u00d728 input',          C.muted],
      [OX  + 14*CS, '28\u00d728 ReLU(conv)',     C.primary],
      [PX  +  7*PS, '14\u00d714 max pool \u2193\u00bd', C.secondary],
    ].forEach(([cx, lbl, col]) =>
      svg.append('text').attr('x', cx).attr('y', IY - 10)
        .attr('text-anchor','middle').attr('fill', col).attr('font-size', 10)
        .attr('font-family','monospace').text(lbl)
    );

    // ── Input grid ────────────────────────────────────────────────────────────
    PIXEL_DATA.forEach((val, idx) => {
      const row = Math.floor(idx/28), col = idx%28;
      const b = Math.round(val*255);
      svg.append('rect')
        .attr('x', IX+col*CS).attr('y', IY+row*CS)
        .attr('width', CS-0.5).attr('height', CS-0.5)
        .attr('fill', `rgb(${b},${b},${b})`)
        .attr('opacity', 0)
        .transition().delay(idx * 0.18).duration(250).attr('opacity', 1);
    });

    // Grid outline
    svg.append('rect').attr('x',IX).attr('y',IY).attr('width',28*CS).attr('height',GH)
      .attr('fill','none').attr('stroke','rgba(255,255,255,0.10)').attr('stroke-width',0.5);

    // ── Kernel display ────────────────────────────────────────────────────────
    const kernG = svg.append('g').attr('opacity', 0);
    kernG.transition().delay(450).duration(300).attr('opacity', 1);

    kernG.append('text').attr('x', KX+24).attr('y', KY-7)
      .attr('text-anchor','middle').attr('fill',C.muted).attr('font-size',9)
      .attr('font-family','monospace').text('kernel');

    KERN.forEach((row, kr) => row.forEach((val, kc) => {
      const ab = Math.abs(val)/2;
      const fc = val < 0 ? `rgba(157,78,221,${0.25+ab*0.6})`
               : val > 0 ? `rgba(0,217,255,${0.25+ab*0.6})`
               : 'rgba(255,255,255,0.06)';
      kernG.append('rect')
        .attr('x', KX+kc*KS).attr('y', KY+kr*KS)
        .attr('width', KS-1).attr('height', KS-1)
        .attr('fill', fc).attr('stroke','rgba(255,255,255,0.2)').attr('stroke-width',0.5);
      kernG.append('text')
        .attr('x', KX+kc*KS+KS/2).attr('y', KY+kr*KS+KS*0.66)
        .attr('text-anchor','middle').attr('font-size',8)
        .attr('fill','rgba(255,255,255,0.9)').attr('font-family','monospace')
        .text(KDISP[kr][kc]);
    }));
    kernG.append('text')
      .attr('x', KX-9).attr('y', IY+GH/2+5)
      .attr('text-anchor','middle').attr('font-size',18).attr('fill',C.muted).text('\u00d7');

    // ── Conv output grid (background cells) ───────────────────────────────────
    const outCells = [];
    for (let r = 0; r < 28; r++)
      for (let c = 0; c < 28; c++)
        outCells.push(svg.append('rect')
          .attr('x', OX+c*CS).attr('y', OY+r*CS)
          .attr('width', CS-0.5).attr('height', CS-0.5)
          .attr('fill','rgba(255,255,255,0.02)')
          .attr('stroke','rgba(255,255,255,0.04)').attr('stroke-width',0.3));

    svg.append('rect').attr('x',OX).attr('y',OY).attr('width',28*CS).attr('height',GH)
      .attr('fill','none').attr('stroke','rgba(0,217,255,0.15)').attr('stroke-width',0.5);

    // ── Scan bar on input ─────────────────────────────────────────────────────
    const SCAN_T0 = 600, ROW_MS = 20;

    const scanBar = svg.append('rect')
      .attr('x', IX).attr('y', IY).attr('width', 28*CS).attr('height', CS*3)
      .attr('fill','rgba(0,217,255,0.10)').attr('stroke',C.primary)
      .attr('stroke-width',0.8).attr('rx',1).attr('opacity',0);
    scanBar.transition().delay(SCAN_T0).duration(100).attr('opacity',1)
      .transition().duration(28*ROW_MS).attr('y', IY+(28-3)*CS)
      .transition().duration(250).attr('opacity',0);

    // ── Conv output cells fill row-by-row ─────────────────────────────────────
    outCells.forEach((cell, idx) => {
      const row = Math.floor(idx/28);
      const val = convOut[idx];
      const b   = Math.round(val*255);
      cell.transition().delay(SCAN_T0 + 60 + row*ROW_MS).duration(ROW_MS + 10)
        .attr('fill', `rgb(${b},${b},${b})`);
    });

    // ── Arrow conv → pool ─────────────────────────────────────────────────────
    const ARR_T  = SCAN_T0 + 100 + 28*ROW_MS + 200;
    const arrMY  = IY + GH/2;
    const arrX1  = OX + 28*CS + 8, arrX2 = PX - 8;
    svg.append('line')
      .attr('x1',arrX1).attr('y1',arrMY).attr('x2',arrX2).attr('y2',arrMY)
      .attr('stroke',C.secondary).attr('stroke-width',1.5)
      .attr('marker-end','url(#arr-cfd)').attr('opacity',0)
      .transition().delay(ARR_T).duration(300).attr('opacity',0.8);
    svg.append('text')
      .attr('x',(arrX1+arrX2)/2).attr('y',arrMY-6)
      .attr('text-anchor','middle').attr('fill',C.muted).attr('font-size',9)
      .attr('font-family','monospace').attr('opacity',0)
      .text('max pool 2\u00d72, stride 2')
      .transition().delay(ARR_T).duration(300).attr('opacity',1);

    // ── Pool scan bar + cells ─────────────────────────────────────────────────
    const POOL_T = ARR_T + 180;
    const POOL_ROW_MS = 22;

    const poolCells = [];
    for (let r = 0; r < 14; r++)
      for (let c = 0; c < 14; c++)
        poolCells.push(svg.append('rect')
          .attr('x', PX+c*PS).attr('y', PY+r*PS)
          .attr('width', PS-1).attr('height', PS-1)
          .attr('fill','rgba(255,255,255,0.02)')
          .attr('stroke','rgba(157,78,221,0.10)').attr('stroke-width',0.4)
          .attr('opacity',0));

    svg.append('rect').attr('x',PX).attr('y',PY).attr('width',14*PS).attr('height',14*PS)
      .attr('fill','none').attr('stroke','rgba(157,78,221,0.25)').attr('stroke-width',0.5)
      .attr('opacity',0).transition().delay(POOL_T-100).duration(300).attr('opacity',1);

    const poolScan = svg.append('rect')
      .attr('x', PX).attr('y', PY).attr('width', 14*PS).attr('height', PS*2)
      .attr('fill','rgba(157,78,221,0.10)').attr('stroke',C.secondary)
      .attr('stroke-width',0.8).attr('rx',1).attr('opacity',0);
    poolScan.transition().delay(POOL_T).duration(100).attr('opacity',1)
      .transition().duration(14*POOL_ROW_MS).attr('y', PY+(14-2)*PS)
      .transition().duration(250).attr('opacity',0);

    poolCells.forEach((cell, idx) => {
      const row = Math.floor(idx/14);
      const val = maxPool[idx];
      const b   = Math.round(val*255);
      cell.transition().delay(POOL_T).duration(50).attr('opacity',1)
        .transition().delay(row*POOL_ROW_MS).duration(POOL_ROW_MS + 10)
        .attr('fill', `rgb(${b},${b},${b})`);
    });

    // ── Footer ────────────────────────────────────────────────────────────────
    const FOOT_T = POOL_T + 100 + 14*POOL_ROW_MS + 300;
    svg.append('text').attr('x', W/2).attr('y', H-6)
      .attr('text-anchor','middle').attr('fill',C.muted).attr('font-size',11)
      .attr('font-style','italic').attr('font-family','monospace').attr('opacity',0)
      .text('Sobel-X edge detector — ReLU zeros negatives; 2\u00d72 max pool halves spatial resolution to 14\u00d714')
      .transition().delay(FOOT_T).duration(400).attr('opacity',1);
  }

  // ── SLIDE: Convolutional Filters — dragging masks across MNIST ──────────
  function initConvFilters(el) {
    const W = 1100, H = 505;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%');

    const CS = 8;
    const IX = 20, IY = Math.round((H - 28*CS) / 2);
    const KX = 268, KS = 16;
    const FMX = 345, FMCS = 4;
    const FM_H = 26 * FMCS;
    const TXX = FMX + 26*FMCS + 18;
    const ROW_Y = [22, 177, 332];

    const FILTERS = [
      { name: 'Vertical Edge', color: C.primary,
        sub: 'detects left \u2194 right contrast',
        kernel: [[-1,0,1],[-2,0,2],[-1,0,1]],
        disp:   [['-1','0','+1'],['-2','0','+2'],['-1','0','+1']] },
      { name: 'Horizontal Edge', color: C.secondary,
        sub: 'detects top \u2195 bottom contrast',
        kernel: [[-1,-2,-1],[0,0,0],[1,2,1]],
        disp:   [['-1','-2','-1'],['0','0','0'],['+1','+2','+1']] },
      { name: 'Blur / Average', color: '#4ade80',
        sub: 'smooths noise \u2014 averages 3\u00d73 neighbours',
        kernel: [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
        disp:   [['1/9','1/9','1/9'],['1/9','1/9','1/9'],['1/9','1/9','1/9']] },
    ];

    function convolve(kernel) {
      const raw = [];
      for (let r = 0; r < 26; r++)
        for (let c = 0; c < 26; c++) {
          let v = 0;
          for (let kr = 0; kr < 3; kr++)
            for (let kc = 0; kc < 3; kc++)
              v += PIXEL_DATA[(r+kr)*28+(c+kc)] * kernel[kr][kc];
          raw.push(v);
        }
      let mn = Infinity, mx = -Infinity;
      for (const v of raw) { if (v < mn) mn = v; if (v > mx) mx = v; }
      return raw.map(v => mx === mn ? 0 : (v - mn) / (mx - mn));
    }

    const featureMaps = FILTERS.map(f => convolve(f.kernel));

    // ── MNIST image ──────────────────────────────────────────────────────
    const imgG = svg.append('g').attr('opacity', 0);
    imgG.transition().duration(500).attr('opacity', 1);

    imgG.append('text')
      .attr('x', IX + 28*CS/2).attr('y', IY - 10)
      .attr('fill', C.muted).attr('font-size', 11).attr('text-anchor', 'middle')
      .attr('font-family', 'monospace').text('28 \u00d7 28 input');

    imgG.append('text')
      .attr('x', IX + 28*CS/2).attr('y', IY + 28*CS + 16)
      .attr('fill', C.muted).attr('font-size', 10).attr('text-anchor', 'middle')
      .attr('font-family', 'monospace').text('digit: ' + MNIST_LABEL);

    for (let r = 0; r < 28; r++)
      for (let c = 0; c < 28; c++) {
        const v = PIXEL_DATA[r*28+c];
        const b = Math.round(v*255);
        imgG.append('rect')
          .attr('x', IX+c*CS).attr('y', IY+r*CS)
          .attr('width', CS-0.5).attr('height', CS-0.5)
          .attr('fill', `rgb(${b},${b},${b})`);
      }

    imgG.append('line')
      .attr('x1', IX+28*CS+12).attr('x2', IX+28*CS+12)
      .attr('y1', 10).attr('y2', H-10)
      .attr('stroke', 'rgba(255,255,255,0.1)').attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,4');

    // ── Scan window ──────────────────────────────────────────────────────
    const scanBox = svg.append('rect')
      .attr('width', 3*CS-0.5).attr('height', 3*CS-0.5)
      .attr('fill', 'none').attr('stroke-width', 2.5).attr('opacity', 0);

    // ── Filter rows ──────────────────────────────────────────────────────
    const fmRects = [[], [], []];

    FILTERS.forEach((f, fi) => {
      const ry  = ROW_Y[fi];
      const fmY = ry + 22;
      const kY  = fmY + Math.round((FM_H - 3*KS) / 2);
      const dy  = 600 + fi * 260;

      const g = svg.append('g').attr('opacity', 0);
      g.transition().delay(dy).duration(300).attr('opacity', 1);

      g.append('text').attr('x', KX).attr('y', ry + 15)
        .attr('fill', f.color).attr('font-size', 12).attr('font-weight', 'bold')
        .attr('font-family', 'monospace').text(f.name + ' Filter');

      f.kernel.forEach((row, kr) => row.forEach((val, kc) => {
        const ab = Math.abs(val) / 2;
        const fc = val < -0.01 ? `rgba(157,78,221,${0.2+ab*0.7})`
                 : val > 0.01  ? `rgba(0,217,255,${0.2+ab*0.7})`
                 :                'rgba(255,255,255,0.05)';
        g.append('rect')
          .attr('x', KX+kc*KS).attr('y', kY+kr*KS)
          .attr('width', KS-1).attr('height', KS-1)
          .attr('fill', fc)
          .attr('stroke', 'rgba(255,255,255,0.2)').attr('stroke-width', 0.5);
        g.append('text')
          .attr('x', KX+kc*KS+KS/2).attr('y', kY+kr*KS+KS*0.66)
          .attr('text-anchor', 'middle').attr('font-size', 8.5)
          .attr('fill', 'rgba(255,255,255,0.9)').attr('font-family', 'monospace')
          .text(f.disp[kr][kc]);
      }));

      g.append('text')
        .attr('x', KX+3*KS+8).attr('y', kY+KS*1.5+5)
        .attr('fill', f.color).attr('font-size', 16).text('\u2192');

      g.append('rect')
        .attr('x', FMX).attr('y', fmY)
        .attr('width', 26*FMCS).attr('height', FM_H)
        .attr('fill', 'rgba(255,255,255,0.02)')
        .attr('stroke', 'rgba(255,255,255,0.08)').attr('stroke-width', 0.5);

      g.append('text').attr('x', FMX+26*FMCS/2).attr('y', fmY-4)
        .attr('fill', C.muted).attr('font-size', 8.5).attr('text-anchor', 'middle')
        .attr('font-family', 'monospace').text('26 \u00d7 26 feature map');

      for (let i = 0; i < 676; i++) {
        const pr = Math.floor(i/26), pc = i%26;
        fmRects[fi].push(
          g.append('rect')
            .attr('x', FMX+pc*FMCS).attr('y', fmY+pr*FMCS)
            .attr('width', FMCS-0.2).attr('height', FMCS-0.2)
            .attr('fill', 'rgba(255,255,255,0)')
        );
      }

      g.append('text').attr('x', TXX).attr('y', fmY + FM_H/2)
        .attr('dominant-baseline', 'middle').attr('fill', C.muted)
        .attr('font-size', 11.5).attr('font-family', 'monospace')
        .text(f.sub);
    });

    const phLbl = svg.append('text')
      .attr('x', W/2).attr('y', H - 8).attr('text-anchor', 'middle')
      .attr('fill', 'rgba(0,0,0,0)').attr('font-size', 11)
      .attr('font-family', 'monospace');

    // ── Chained scan animations ──────────────────────────────────────────
    function runScan(fi, cb) {
      const fmap = featureMaps[fi];
      const MS = 1100;
      let done = 0;
      scanBox.attr('stroke', FILTERS[fi].color).attr('opacity', 1);
      const timer = d3.timer(elapsed => {
        const pos = Math.min(Math.floor(elapsed/MS * 676), 676);
        for (let i = done; i < pos; i++) {
          const b = Math.round(fmap[i]*255);
          fmRects[fi][i].attr('fill', `rgb(${b},${b},${b})`);
        }
        done = pos;
        const sp = Math.min(pos, 675);
        scanBox.attr('x', IX+(sp%26)*CS).attr('y', IY+Math.floor(sp/26)*CS);
        if (elapsed >= MS) {
          for (let i = done; i < 676; i++) {
            const b = Math.round(fmap[i]*255);
            fmRects[fi][i].attr('fill', `rgb(${b},${b},${b})`);
          }
          timer.stop();
          scanBox.attr('opacity', 0);
          if (cb) setTimeout(cb, 200);
        }
      });
    }

    const startDelay = 600 + 2*260 + 350;
    setTimeout(() => {
      runScan(0, () =>
        runScan(1, () =>
          runScan(2, () => {
            phLbl
              .text('CNNs learn many such filters \u2014 stacked layers detect increasingly complex patterns')
              .transition().duration(400).attr('fill', C.muted);
          })
        )
      );
    }, startDelay);
  }

  // ── SLIDE: CNN Parameters & Learned Filters ─────────────────────────────
  function initCNNParams(el) {
    const W = 1100, H = 505;
    const svg = d3.select(el).append('svg')
      .attr('viewBox',`0 0 ${W} ${H}`)
      .style('width','100%').style('height','100%');

    // Kernel colour: dark bg → red (positive) / blue (negative)
    const kColor = v => {
      if (v >= 0) return `rgb(${~~(v*218+26*(1-v))},${~~(v*55+26*(1-v))},${~~(v*48+46*(1-v))})`;
      const t = -v;
      return `rgb(${~~(t*48+26*(1-t))},${~~(t*140+26*(1-t))},${~~(t*218+46*(1-t))})`;
    };

    // ── Parameter table (left) ────────────────────────────────────────────────
    const TX = 28, TY0 = 22, ROW_H = 40, BAR_MAX = 110, MAX_P = 100480;
    const ROWS = [
      { layer:'Conv2D (1)',  formula:'16×(1×5×5) + 16',  params:416,    col:C.primary   },
      { layer:'MaxPool',     formula:'—',                 params:0,      col:C.muted     },
      { layer:'Conv2D (2)',  formula:'16×(16×3×3) + 16', params:2320,   col:C.primary   },
      { layer:'MaxPool',     formula:'—',                 params:0,      col:C.muted     },
      { layer:'FC (1)',      formula:'784 × 128 + 128',   params:100480, col:'#4ade80'   },
      { layer:'FC (2)',      formula:'128 × 10 + 10',     params:1290,   col:'#4ade80'   },
    ];
    const TOTAL = ROWS.reduce((s,r)=>s+r.params, 0);   // 104 506

    svg.append('text').attr('x',TX).attr('y',TY0)
      .attr('fill',C.text).attr('font-size',11).attr('font-family','monospace')
      .attr('font-weight','bold').text('Trainable Parameters');

    // Column headers
    const hg = svg.append('g').attr('opacity',0);
    hg.transition().delay(60).duration(250).attr('opacity',1);
    [['Layer',TX],['Formula',TX+132],['Count',TX+310],['Share',TX+372]]
      .forEach(([t,x]) => hg.append('text').attr('x',x).attr('y',TY0+20)
        .attr('fill',C.muted).attr('font-size',9).attr('font-family','monospace')
        .attr('font-weight','bold').text(t));
    hg.append('line').attr('x1',TX).attr('y1',TY0+27).attr('x2',TX+492).attr('y2',TY0+27)
      .attr('stroke',C.muted).attr('stroke-width',0.7).attr('opacity',0.3);

    ROWS.forEach((r,i) => {
      const y = TY0 + 42 + i*ROW_H;
      const g = svg.append('g').attr('opacity',0);
      g.transition().delay(120+i*100).duration(240).attr('opacity',1);

      g.append('text').attr('x',TX).attr('y',y)
        .attr('fill',r.col).attr('font-size',10).attr('font-family','monospace')
        .attr('font-weight','bold').text(r.layer);
      g.append('text').attr('x',TX+132).attr('y',y)
        .attr('fill', r.params>0 ? C.text : C.muted)
        .attr('font-size',9.5).attr('font-family','monospace').text(r.formula);
      g.append('text').attr('x',TX+358).attr('y',y)
        .attr('fill', r.params>0 ? C.text : C.muted)
        .attr('font-size',10).attr('font-family','monospace').attr('text-anchor','end')
        .text(r.params>0 ? r.params.toLocaleString() : '—');

      if (r.params > 0)
        g.append('rect').attr('x',TX+370).attr('y',y-11).attr('height',13).attr('width',0).attr('rx',2)
          .attr('fill',r.col).attr('opacity',0.62)
          .transition().delay(120+i*100+180).duration(380)
          .attr('width', Math.max(2, (r.params/MAX_P)*BAR_MAX));

      g.append('line').attr('x1',TX).attr('y1',y+9).attr('x2',TX+492).attr('y2',y+9)
        .attr('stroke',C.muted).attr('stroke-width',0.25).attr('opacity',0.18);
    });

    // Total row
    const totalY = TY0 + 42 + ROWS.length*ROW_H + 10;
    const tg = svg.append('g').attr('opacity',0);
    tg.transition().delay(120+ROWS.length*100+60).duration(280).attr('opacity',1);
    tg.append('line').attr('x1',TX).attr('y1',totalY-16).attr('x2',TX+492).attr('y2',totalY-16)
      .attr('stroke',C.muted).attr('stroke-width',0.8).attr('opacity',0.42);
    tg.append('text').attr('x',TX).attr('y',totalY)
      .attr('fill',C.accent).attr('font-size',11).attr('font-family','monospace')
      .attr('font-weight','bold').text('Total');
    tg.append('text').attr('x',TX+358).attr('y',totalY)
      .attr('fill',C.accent).attr('font-size',11).attr('font-family','monospace')
      .attr('font-weight','bold').attr('text-anchor','end').text(TOTAL.toLocaleString());
    tg.append('rect').attr('x',TX+370).attr('y',totalY-13).attr('height',15).attr('width',0).attr('rx',2)
      .attr('fill',C.accent).attr('opacity',0.65)
      .transition().delay(120+ROWS.length*100+260).duration(460).attr('width',BAR_MAX);

    // Vertical divider
    svg.append('line').attr('x1',548).attr('y1',8).attr('x2',548).attr('y2',H-8)
      .attr('stroke',C.muted).attr('stroke-width',0.5).attr('opacity',0.2);

    // ── Conv1 kernels (right) ─────────────────────────────────────────────────
    const KX0=562, KY0=22, KSIZE=5, KS=15, NCOLS=4, CELL_W=133, CELL_H=112;

    svg.append('text').attr('x',KX0).attr('y',KY0)
      .attr('fill',C.primary).attr('font-size',11).attr('font-family','monospace')
      .attr('font-weight','bold').text('Conv1 Learned Filters  (5 × 5)');
    svg.append('text').attr('x',KX0).attr('y',KY0+14)
      .attr('fill',C.muted).attr('font-size',8).attr('font-family','monospace')
      .text('red = positive weight  ·  blue = negative weight');

    // Actual trained conv1 kernels from cnn-weights.js
    const KERNELS = (window.CNN_WEIGHTS && window.CNN_WEIGHTS.conv1) || [];
    KERNELS.forEach((kern, ki) => {
      const col = ki % NCOLS, row = Math.floor(ki / NCOLS);
      const cx = KX0 + col*CELL_W, cy = KY0 + 26 + row*CELL_H;

      const g = svg.append('g').attr('opacity',0);
      g.transition().delay(180 + ki*48).duration(220).attr('opacity',1);

      const flat = kern.flat(), vmax = Math.max(...flat.map(Math.abs)) || 1;

      g.append('text').attr('x', cx + KSIZE*KS/2).attr('y', cy+9)
        .attr('text-anchor','middle').attr('fill',C.muted)
        .attr('font-size',7.5).attr('font-family','monospace').text(`F${ki+1}`);

      for (let r=0; r<KSIZE; r++)
        for (let c=0; c<KSIZE; c++) {
          const v = kern[r][c] / vmax;
          g.append('rect')
            .attr('x', cx + c*KS).attr('y', cy+13 + r*KS)
            .attr('width', KS-1).attr('height', KS-1).attr('rx',1)
            .attr('fill', kColor(v));
        }
    });
  }

  // ── SLIDE: CNN slab viewer — 4×4 grid of feature maps for one layer ─────────
  function initCNNSlabs(el, fmKey, title) {
    const W=1100, H=505;
    const svg = d3.select(el).append('svg')
      .attr('viewBox',`0 0 ${W} ${H}`)
      .style('width','100%').style('height','100%');

    const s = window.CNN_SAMPLE || null;
    if (!s) {
      svg.append('text').attr('x',W/2).attr('y',H/2).attr('text-anchor','middle')
        .attr('fill',C.muted).attr('font-family','monospace').text('cnn-weights.js not loaded');
      return;
    }

    const fmaps = s['fm_' + fmKey];
    const FN = Math.round(Math.sqrt(fmaps[0].length));

    // Header
    svg.append('text').attr('x',28).attr('y',30)
      .attr('fill',C.text).attr('font-size',20).attr('font-family','monospace')
      .attr('font-weight','bold').text(title);
    svg.append('text').attr('x',28).attr('y',46)
      .attr('fill',C.muted).attr('font-size',10.5).attr('font-family','monospace')
      .text(`${FN}\u00d7${FN} maps  \u00b7  16 channels  \u00b7  digit: ${s.label}  \u2192  predicted: ${s.predicted}`);

    // Layout: 4 cols × 4 rows
    const COLS=4, ROWS=4, padX=28, padY=57, gapX=14, gapY=10;
    const cellW = (W - 2*padX - (COLS-1)*gapX) / COLS;
    const cellH = (H - padY - 12 - (ROWS-1)*gapY) / ROWS;
    const mapSz = Math.min(cellH - 16, cellW - 18);
    const px = mapSz / FN;

    // Layer colour accent matching architecture slide
    const isConv = fmKey==='c1'||fmKey==='c2';
    const accent = isConv ? C.primary : C.secondary;

    fmaps.forEach((fm, fi) => {
      const col = fi % COLS, row = Math.floor(fi / COLS);
      const cx = padX + col * (cellW + gapX);
      const cy = padY + row * (cellH + gapY);
      const mx = cx + (cellW - mapSz) / 2;

      const g = svg.append('g').attr('opacity', 0);
      g.transition().delay(30 + fi * 28).duration(220).attr('opacity', 1);

      // Grayscale feature map
      for (let r=0; r<FN; r++) for (let c=0; c<FN; c++) {
        const b = Math.round(fm[r*FN+c] * 255);
        g.append('rect')
          .attr('x', mx + c*px).attr('y', cy + r*px)
          .attr('width', px+0.5).attr('height', px+0.5)
          .attr('fill', `rgb(${b},${b},${b})`);
      }

      // Outline
      g.append('rect').attr('x',mx).attr('y',cy)
        .attr('width',mapSz).attr('height',mapSz)
        .attr('fill','none').attr('stroke','rgba(255,255,255,0.32)').attr('stroke-width',0.8);

      // Channel label
      g.append('text').attr('x',mx + mapSz/2).attr('y',cy + mapSz + 11)
        .attr('text-anchor','middle').attr('fill',C.muted)
        .attr('font-size',9).attr('font-family','monospace').text(`ch ${fi+1}`);
    });

    // Bottom accent line
    const lineG = svg.append('g').attr('opacity',0);
    lineG.transition().delay(30+16*28+80).duration(300).attr('opacity',1);
    lineG.append('line').attr('x1',padX).attr('y1',H-14).attr('x2',W-padX).attr('y2',H-14)
      .attr('stroke',accent).attr('stroke-width',1.5).attr('opacity',0.3);
    lineG.append('text').attr('x',W/2).attr('y',H-5).attr('text-anchor','middle')
      .attr('fill',accent).attr('font-size',9).attr('font-family','monospace')
      .text(isConv ? 'ReLU activation — brighter = stronger response' : 'max-pooled activations — retains strongest responses');
  }

  // ── SLIDE: CNN Evaluation — digit classification with cycling samples ────────
  function initCNNEval(el) {
    // Reset to the same sample CNN_SAMPLE uses (random per page load),
    // so revisiting the slide restarts like all other CNN slides.
    const samples = window.CNN_WEIGHTS && window.CNN_WEIGHTS.samples;
    const cnnSample = window.CNN_SAMPLE;
    if (samples && cnnSample) {
      const i = samples.findIndex(s => s.label === cnnSample.label);
      el._cnnEvalIdx = i >= 0 ? i : 0;
    } else {
      el._cnnEvalIdx = 0;
    }
    _renderCNNEval(el);
  }

  function _renderCNNEval(el) {
    el.innerHTML = '';
    const W = 1100, H = 505;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width', '100%').style('height', '100%');

    const samples = window.CNN_WEIGHTS && window.CNN_WEIGHTS.samples;
    if (!samples || !samples.length) {
      svg.append('text').attr('x', W/2).attr('y', H/2).attr('text-anchor', 'middle')
        .attr('fill', C.muted).attr('font-family', 'monospace').text('cnn-weights.js not loaded');
      return;
    }

    const idx = el._cnnEvalIdx || 0;
    const s   = samples[idx];

    addArrowDef(svg, 'arr-cnn-eval', C.primary);

    // ── Left panel: MNIST pixel grid ─────────────────────────────────────────
    const CELL = 8, GAP = 1;
    const GW = 28 * CELL, GH = 28 * CELL;
    const GX = 12, GY = (H - GH) / 2;

    s.pixels.forEach((val, pidx) => {
      const row = Math.floor(pidx / 28), col = pidx % 28;
      const g   = Math.round(val * 255);
      svg.append('rect')
        .attr('x', GX + col * CELL + GAP).attr('y', GY + row * CELL + GAP)
        .attr('width', CELL - GAP * 2).attr('height', CELL - GAP * 2)
        .attr('fill', `rgb(${g},${g},${g})`)
        .attr('opacity', 0)
        .transition().delay(pidx * 0.12).duration(200).attr('opacity', 1);
    });

    svg.append('text').attr('x', GX + GW / 2).attr('y', GY - 8)
      .attr('text-anchor', 'middle').attr('fill', C.primary).attr('font-size', 12)
      .text(`28 × 28  (label: ${s.label})`);

    // Arrow grid → output
    const ARR_X1 = GX + GW + 8, ARR_X2 = GX + GW + 40, ARR_Y = H / 2;
    svg.append('line')
      .attr('x1', ARR_X1).attr('y1', ARR_Y).attr('x2', ARR_X2).attr('y2', ARR_Y)
      .attr('stroke', C.primary).attr('stroke-width', 1.5)
      .attr('marker-end', 'url(#arr-cnn-eval)')
      .attr('opacity', 0)
      .transition().delay(500).duration(300).attr('opacity', 0.6);

    // ── Right panel: output nodes + probability bars ──────────────────────────
    const probs     = s.probs;
    const predicted = s.predicted;
    const R = 24;
    const SPACING = H / 10;
    const NY = (i) => SPACING / 2 + i * SPACING;
    const NX = GX + GW + 58;
    const BAR_X = NX + R + 16;
    const MAX_BAR = 340;

    svg.append('text').attr('x', NX).attr('y', 20)
      .attr('text-anchor', 'middle').attr('fill', C.primary)
      .attr('font-size', 13).attr('font-weight', 'bold')
      .text('Output Layer — 10 classes');

    for (let i = 0; i < 10; i++) {
      const cy    = NY(i);
      const isMax = i === predicted;
      const col   = isMax ? C.accent : C.primary;

      svg.append('circle')
        .attr('cx', NX).attr('cy', cy).attr('r', R)
        .attr('fill', isMax ? 'rgba(255,0,110,0.18)' : C.nodeFill)
        .attr('stroke', col).attr('stroke-width', isMax ? 2 : 1)
        .attr('opacity', 0)
        .transition().delay(i * 80).duration(350).attr('opacity', 1);

      svg.append('text').attr('x', NX).attr('y', cy + 5)
        .attr('text-anchor', 'middle').attr('fill', col)
        .attr('font-size', 15).attr('font-weight', isMax ? 'bold' : 'normal')
        .attr('opacity', 0).text(String(i))
        .transition().delay(i * 80).duration(350).attr('opacity', 1);

      svg.append('rect')
        .attr('x', BAR_X).attr('y', cy - 10)
        .attr('width', MAX_BAR).attr('height', 20)
        .attr('fill', 'rgba(255,255,255,0.04)').attr('rx', 3)
        .attr('opacity', 0)
        .transition().delay(i * 80 + 200).duration(300).attr('opacity', 1);

      const barW = Math.pow(probs[i], 0.3) * MAX_BAR;
      svg.append('rect')
        .attr('x', BAR_X).attr('y', cy - 10)
        .attr('width', 0).attr('height', 20)
        .attr('fill', isMax ? C.accent : 'rgba(0,217,255,0.45)').attr('rx', 3)
        .transition().delay(i * 80 + 400).duration(600).attr('width', barW);

      const p = probs[i];
      const probLabel = p >= 0.0001
        ? (p * 100).toFixed(p >= 0.01 ? 1 : 3) + '%'
        : p.toExponential(1);
      svg.append('text')
        .attr('x', BAR_X + barW + 5).attr('y', cy + 5)
        .attr('fill', isMax ? C.accent : C.muted)
        .attr('font-size', 11).attr('font-family', 'monospace')
        .attr('font-weight', isMax ? 'bold' : 'normal')
        .attr('opacity', 0).text(probLabel)
        .transition().delay(i * 80 + 700).duration(300).attr('opacity', 1);
    }

    // "Predicted: N" badge
    const badgeX = BAR_X + MAX_BAR + 18;
    const badgeG = svg.append('g').attr('opacity', 0);
    badgeG.append('rect')
      .attr('x', badgeX).attr('y', NY(predicted) - 38)
      .attr('width', 128).attr('height', 88)
      .attr('fill', 'rgba(255,0,110,0.1)').attr('stroke', C.accent)
      .attr('stroke-width', 2).attr('rx', 10);
    badgeG.append('text')
      .attr('x', badgeX + 64).attr('y', NY(predicted) - 16)
      .attr('text-anchor', 'middle').attr('fill', C.accent)
      .attr('font-size', 12).text('Predicted');
    badgeG.append('text')
      .attr('x', badgeX + 64).attr('y', NY(predicted) + 22)
      .attr('text-anchor', 'middle').attr('fill', C.accent)
      .attr('font-size', 30).attr('font-weight', 'bold').text(String(predicted));
    const correct = s.label === s.predicted;
    badgeG.append('text')
      .attr('x', badgeX + 64).attr('y', NY(predicted) + 42)
      .attr('text-anchor', 'middle')
      .attr('fill', correct ? '#4ade80' : '#f87171').attr('font-size', 9)
      .attr('font-family', 'monospace').text(correct ? '\u2713 correct' : `\u2717 actual: ${s.label}`);
    badgeG.transition().delay(1200).duration(500).attr('opacity', 1);

    // Footer note
    svg.append('text').attr('x', BAR_X + MAX_BAR / 2).attr('y', H - 6)
      .attr('text-anchor', 'middle').attr('fill', C.muted).attr('font-size', 11)
      .attr('font-style', 'italic').attr('opacity', 0)
      .text('real CNN predictions (Conv2D\u00d72 + MaxPool\u00d72 + FC128 + FC10, ~99% test accuracy) \u2014 bar scale: p\u00b00\u00b73')
      .transition().delay(1400).duration(400).attr('opacity', 1);

    // "Next sample →" cycling button
    const btnG = svg.append('g').attr('cursor', 'pointer').attr('opacity', 0);
    btnG.transition().delay(1600).duration(300).attr('opacity', 1);
    const btnX = GX, btnY = H - 42, btnW = 160, btnH = 28;
    btnG.append('rect')
      .attr('x', btnX).attr('y', btnY).attr('width', btnW).attr('height', btnH)
      .attr('fill', 'rgba(0,217,255,0.08)').attr('stroke', C.primary)
      .attr('stroke-width', 1).attr('rx', 5);
    btnG.append('text')
      .attr('x', btnX + btnW / 2).attr('y', btnY + 18)
      .attr('text-anchor', 'middle').attr('fill', C.primary)
      .attr('font-size', 12).attr('font-family', 'monospace')
      .text(`Next sample \u2192  (${idx + 1} / ${samples.length})`);
    btnG.on('click', () => {
      el._cnnEvalIdx = (idx + 1) % samples.length;
      _renderCNNEval(el);
    });
  }

  // ── SLIDE: Encoder–Decoder architecture animation ────────────────────────
  function initFCNAnimation(el) {
    const W = 1100, H = 505;
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .style('width','100%').style('height','100%');

    const OA = Math.PI / 6, OS2 = 0.48;
    const oX = d => d * Math.cos(OA) * OS2;
    const oY = d => -d * Math.sin(OA) * OS2;

    const CY = 258, FW = 32, GAP = 16;

    const ENC = { f:'rgba(0,145,192,0.52)',  t:'rgba(0,178,220,0.68)',  s:'rgba(0,95,142,0.52)'  };
    const DEC = { f:'rgba(126,50,206,0.52)', t:'rgba(155,76,240,0.68)', s:'rgba(80,26,150,0.52)' };
    const BOT = { f:'rgba(0,55,135,0.78)',   t:'rgba(0,85,175,0.88)',   s:'rgba(0,30,85,0.78)'   };
    const INP = { t:'rgba(195,198,210,0.62)',s:'rgba(100,103,115,0.48)'};
    const OUT = { t:'rgba(50,182,80,0.70)',  s:'rgba(16,96,35,0.55)'   };

    // Bottleneck is 1×1 spatial — very thin face, very deep (many channels)
    const STAGES = [
      { label:'Input',       sub:'224×224',      fh:176, depth:6,  sk:null, col:INP, role:'input'  },
      { label:'Conv + Pool', sub:'112×112 × 64', fh:92,  depth:24, sk:0,   col:ENC, role:'enc'    },
      { label:'Conv + Pool', sub:'56×56 × 128',  fh:50,  depth:38, sk:1,   col:ENC, role:'enc'    },
      { label:'Bottleneck',    sub:'1×1 × 4096',   fh:6,   depth:90, sk:null, col:BOT, role:'bot'   },
      { label:'Unpool+Deconv', sub:'56×56 × 128',  fh:50,  depth:38, sk:1,   col:DEC, role:'dec'    },
      { label:'Unpool+Deconv', sub:'112×112 × 64', fh:92,  depth:24, sk:0,   col:DEC, role:'dec'    },
      { label:'Seg. Map',    sub:'224×224 × C',  fh:176, depth:6,  sk:null, col:OUT, role:'output' },
    ];

    // X layout
    const xs = [];
    let acc = 0;
    for (const s of STAGES) { xs.push(acc); acc += FW + oX(s.depth) + GAP; }
    const startX = Math.round((W - (acc - GAP)) / 2);

    function drawSlab(g, x, y, fh, depth, col) {
      const ox = oX(depth), oy = oY(depth);
      g.append('polygon')
        .attr('points',`${x},${y} ${x+FW},${y} ${x+FW+ox},${y+oy} ${x+ox},${y+oy}`)
        .attr('fill',col.t).attr('stroke','rgba(0,0,0,0.22)').attr('stroke-width',0.6);
      g.append('polygon')
        .attr('points',`${x+FW},${y} ${x+FW+ox},${y+oy} ${x+FW+ox},${y+oy+fh} ${x+FW},${y+fh}`)
        .attr('fill',col.s).attr('stroke','rgba(0,0,0,0.22)').attr('stroke-width',0.6);
      g.append('rect')
        .attr('x',x).attr('y',y).attr('width',FW).attr('height',fh)
        .attr('fill',col.f).attr('stroke','rgba(255,255,255,0.42)').attr('stroke-width',0.8);
    }

    // Landscape scene drawn on input slab face
    function drawLandscape(g, x, y, fh) {
      // Sky
      g.append('rect').attr('x',x).attr('y',y).attr('width',FW).attr('height',fh*0.36).attr('fill','#3a7fc1');
      // Sun
      g.append('circle').attr('cx',x+FW*0.74).attr('cy',y+fh*0.09).attr('r',4.5).attr('fill','#ffe450');
      // Background mountain
      g.append('polygon')
        .attr('points',`${x},${y+fh*0.44} ${x+FW*0.42},${y+fh*0.18} ${x+FW},${y+fh*0.44}`)
        .attr('fill','#9090a8');
      // Foreground mountain
      g.append('polygon')
        .attr('points',`${x},${y+fh*0.50} ${x+FW*0.24},${y+fh*0.30} ${x+FW*0.58},${y+fh*0.50}`)
        .attr('fill','#686078');
      // Treeline
      g.append('rect').attr('x',x).attr('y',y+fh*0.46).attr('width',FW).attr('height',fh*0.09).attr('fill','#2a6038');
      // Ground / field
      g.append('rect').attr('x',x).attr('y',y+fh*0.54).attr('width',FW).attr('height',fh*0.46).attr('fill','#4a7c30');
      // Road (perspective trapezoid)
      g.append('polygon')
        .attr('points',`${x+FW*0.40},${y+fh*0.54} ${x+FW*0.60},${y+fh*0.54} ${x+FW*0.82},${y+fh} ${x+FW*0.18},${y+fh}`)
        .attr('fill','#b0a060');
      // House body
      const hx=x+FW*0.07, hy=y+fh*0.56;
      g.append('rect').attr('x',hx).attr('y',hy+fh*0.035).attr('width',8).attr('height',fh*0.09).attr('fill','#c8a870');
      // House roof
      g.append('polygon')
        .attr('points',`${hx-1},${hy+fh*0.035} ${hx+4},${hy} ${hx+9},${hy+fh*0.035}`)
        .attr('fill','#b04028');
      // Tree trunk
      g.append('rect').attr('x',x+FW*0.79).attr('y',y+fh*0.55).attr('width',2).attr('height',fh*0.07).attr('fill','#6b4020');
      // Tree foliage
      g.append('circle').attr('cx',x+FW*0.80).attr('cy',y+fh*0.53).attr('r',5.5).attr('fill','#246025');
      // Outline
      g.append('rect').attr('x',x).attr('y',y).attr('width',FW).attr('height',fh)
        .attr('fill','none').attr('stroke','rgba(255,255,255,0.62)').attr('stroke-width',1.1);
    }

    // Semantic segmentation map — same layout as landscape, class colours
    function drawSegMap(g, x, y, fh) {
      // Sky class
      g.append('rect').attr('x',x).attr('y',y).attr('width',FW).attr('height',fh*0.36).attr('fill','rgba(74,148,235,0.92)');
      // Mountain class (covers both mountains + sky overlap area)
      g.append('polygon')
        .attr('points',`${x},${y+fh*0.36} ${x+FW*0.42},${y+fh*0.18} ${x+FW},${y+fh*0.36} ${x+FW},${y+fh*0.54} ${x},${y+fh*0.54}`)
        .attr('fill','rgba(175,145,215,0.92)');
      // Vegetation class
      g.append('rect').attr('x',x).attr('y',y+fh*0.46).attr('width',FW).attr('height',fh*0.09).attr('fill','rgba(40,145,68,0.92)');
      // Ground class
      g.append('rect').attr('x',x).attr('y',y+fh*0.54).attr('width',FW).attr('height',fh*0.46).attr('fill','rgba(88,198,88,0.90)');
      // Road class
      g.append('polygon')
        .attr('points',`${x+FW*0.40},${y+fh*0.54} ${x+FW*0.60},${y+fh*0.54} ${x+FW*0.82},${y+fh} ${x+FW*0.18},${y+fh}`)
        .attr('fill','rgba(200,185,110,0.92)');
      // Building class
      g.append('rect').attr('x',x+FW*0.07).attr('y',y+fh*0.595).attr('width',8).attr('height',fh*0.125)
        .attr('fill','rgba(225,95,45,0.92)');
      // Tree class
      g.append('circle').attr('cx',x+FW*0.80).attr('cy',y+fh*0.53).attr('r',5.5).attr('fill','rgba(38,148,52,0.92)');
      // Outline
      g.append('rect').attr('x',x).attr('y',y).attr('width',FW).attr('height',fh)
        .attr('fill','none').attr('stroke','rgba(255,255,255,0.62)').attr('stroke-width',1.1);
    }

    const skipEnc = {}, skipDec = {};

    STAGES.forEach((s, si) => {
      const x = startX + xs[si];
      const y = CY - s.fh / 2;
      const delay = 120 + si * 270;

      const g = svg.append('g').attr('opacity', 0);
      g.transition().delay(delay).duration(280).attr('opacity', 1);

      if (s.role === 'input') {
        const ox2 = oX(s.depth), oy2 = oY(s.depth);
        g.append('polygon')
          .attr('points',`${x},${y} ${x+FW},${y} ${x+FW+ox2},${y+oy2} ${x+ox2},${y+oy2}`)
          .attr('fill',s.col.t).attr('stroke','rgba(0,0,0,0.15)').attr('stroke-width',0.5);
        g.append('polygon')
          .attr('points',`${x+FW},${y} ${x+FW+ox2},${y+oy2} ${x+FW+ox2},${y+oy2+s.fh} ${x+FW},${y+s.fh}`)
          .attr('fill',s.col.s).attr('stroke','rgba(0,0,0,0.15)').attr('stroke-width',0.5);
        drawLandscape(g, x, y, s.fh);

      } else if (s.role === 'output') {
        const ox2 = oX(s.depth), oy2 = oY(s.depth);
        g.append('polygon')
          .attr('points',`${x},${y} ${x+FW},${y} ${x+FW+ox2},${y+oy2} ${x+ox2},${y+oy2}`)
          .attr('fill',s.col.t).attr('stroke','rgba(0,0,0,0.15)').attr('stroke-width',0.5);
        g.append('polygon')
          .attr('points',`${x+FW},${y} ${x+FW+ox2},${y+oy2} ${x+FW+ox2},${y+oy2+s.fh} ${x+FW},${y+s.fh}`)
          .attr('fill',s.col.s).attr('stroke','rgba(0,0,0,0.15)').attr('stroke-width',0.5);
        drawSegMap(g, x, y, s.fh);

      } else if (s.role === 'bot') {
        // Extremely thin face (1×1 spatial) but very deep (512 channels) — depth dominates visually
        drawSlab(g, x, y, s.fh, s.depth, s.col);
        // Small bright dot on the face to emphasise "single pixel"
        g.append('rect')
          .attr('x', x+FW/2-2).attr('y', y+s.fh/2-2)
          .attr('width',4).attr('height',4).attr('rx',1)
          .attr('fill','rgba(180,220,255,0.95)');

      } else {
        drawSlab(g, x, y, s.fh, s.depth, s.col);
      }

      // Label below — use a minimum offset so thin bottleneck label still clears CY
      const lc = s.role==='enc' ? C.primary : s.role==='dec' ? C.secondary
               : s.role==='output' ? '#4ade80' : s.role==='bot' ? 'rgba(110,190,255,0.92)' : C.muted;
      const lx = x + (FW + oX(s.depth)) / 2;
      const ly = CY + Math.max(s.fh, 20) / 2 + 14;
      g.append('text').attr('x',lx).attr('y',ly)
        .attr('text-anchor','middle').attr('font-size',9.5).attr('font-family','monospace')
        .attr('font-weight','bold').attr('fill',lc).text(s.label);
      g.append('text').attr('x',lx).attr('y',ly+12)
        .attr('text-anchor','middle').attr('font-size',7.5).attr('font-family','monospace')
        .attr('fill',C.muted).text(s.sub);

      // Arrow connector
      if (si < STAGES.length - 1) {
        const rx = x + FW + oX(s.depth);
        g.append('text').attr('x',rx+GAP/2-3).attr('y',CY+4)
          .attr('fill',C.muted).attr('font-size',11).text('→');
      }

      // Track skip endpoints
      if (s.role === 'enc' && s.sk !== null) skipEnc[s.sk] = { cx: x+FW/2, top: y };
      if (s.role === 'dec' && s.sk !== null) skipDec[s.sk] = { cx: x+FW/2, top: y };
    });

    // Bottleneck annotation
    const botDelay = 120 + 3*270 + 180;
    const botSt = STAGES[3];
    const botX = startX + xs[3] + (FW + oX(botSt.depth)) / 2;
    svg.append('text').attr('x',botX).attr('y', CY - 18)
      .attr('text-anchor','middle').attr('font-size',9).attr('font-family','monospace')
      .attr('fill','rgba(110,190,255,0.88)').attr('opacity',0)
      .transition().delay(botDelay).duration(300).attr('opacity',1)
      .text('FC layers — 1×1 spatial');

    // "Encoder ↓" / "Decoder ↑" section labels
    const secDelay = 120 + 6*270 + 350;
    const labelY = CY + 92 + 34;
    const encMidX = startX + (xs[1] + xs[2] + FW + oX(STAGES[2].depth)) / 2;
    const decMidX = startX + (xs[4] + xs[5] + FW + oX(STAGES[5].depth)) / 2;
    const secG = svg.append('g').attr('opacity',0);
    secG.transition().delay(secDelay).duration(400).attr('opacity',1);
    secG.append('text').attr('x',encMidX).attr('y',labelY)
      .attr('text-anchor','middle').attr('font-size',11).attr('font-family','monospace')
      .attr('font-weight','bold').attr('fill',C.primary).text('Encoder ↓');
    secG.append('text').attr('x',decMidX).attr('y',labelY)
      .attr('text-anchor','middle').attr('font-size',11).attr('font-family','monospace')
      .attr('font-weight','bold').attr('fill',C.secondary).text('Decoder ↑');

    // Pooling-index (switch variable) connections animate in after bottleneck appears
    const skipDelay = 120 + 3*270 + 400;
    [
      { sk:0, peakY:30, lbl:'pool indices (112²)' },
      { sk:1, peakY:56, lbl:'pool indices (56²)'  },
    ].forEach(({ sk, peakY, lbl }, i) => {
      setTimeout(() => {
        const enc = skipEnc[sk], dec = skipDec[sk];
        if (!enc || !dec) return;
        const pathD = `M ${enc.cx},${enc.top} C ${enc.cx},${peakY} ${dec.cx},${peakY} ${dec.cx},${dec.top}`;
        const p = svg.append('path')
          .attr('d', pathD).attr('fill','none')
          .attr('stroke','rgba(255,160,20,0.82)').attr('stroke-width', sk===0 ? 1.8 : 1.4);
        const len = p.node().getTotalLength();
        p.attr('stroke-dasharray', `${len} ${len}`)
          .attr('stroke-dashoffset', len)
          .transition().duration(650).ease(d3.easeLinear)
          .attr('stroke-dashoffset', 0)
          .on('end', () => p.attr('stroke-dasharray','4,3').attr('stroke-dashoffset',0));
        svg.append('text')
          .attr('x', (enc.cx+dec.cx)/2).attr('y', peakY-7)
          .attr('text-anchor','middle').attr('font-size',8.5).attr('font-family','monospace')
          .attr('fill','rgba(255,165,30,0.92)').attr('opacity',0)
          .transition().delay(660).duration(280).attr('opacity',1)
          .text(lbl);
      }, skipDelay + i*280);
    });
    // "switch variables" header label
    setTimeout(() => {
      const enc0 = skipEnc[0], dec0 = skipDec[0];
      if (!enc0 || !dec0) return;
      svg.append('text')
        .attr('x', (enc0.cx+dec0.cx)/2).attr('y', 14)
        .attr('text-anchor','middle').attr('font-size',8).attr('font-family','monospace')
        .attr('fill','rgba(255,165,30,0.70)').attr('opacity',0)
        .transition().delay(700).duration(300).attr('opacity',1)
        .text('switch variables — only max position stored (not full feature maps)');
    }, skipDelay);
  }

  // ── Slide registration ───────────────────────────────────────────────────
  const SLIDES = {
    'activation-functions': initActivation,
    'backprop-simple':      initBackpropSimple,
    'backprop-network':     initBackpropNetwork,
    'backprop-tensor':      initBackpropTensor,
    'batch-epochs':         initBatchEpochs,
    'bias-backprop':        initBiasBackprop,
    'cnn-architecture':     initCNNArchitecture,
    // 'cnn-fcn-anim' handled by inline VGG-16 animation in presentation.html
    'cnn-eval':             initCNNEval,
    'cnn-params':           initCNNParams,
    'cnn-slabs-c1':         el => initCNNSlabs(el, 'c1', 'Conv2D (1) \u2014 16 Activation Maps'),
    'cnn-slabs-p1':         el => initCNNSlabs(el, 'p1', 'MaxPool (1) \u2014 16 Activation Maps'),
    'cnn-slabs-c2':         el => initCNNSlabs(el, 'c2', 'Conv2D (2) \u2014 16 Activation Maps'),
    'cnn-slabs-p2':         el => initCNNSlabs(el, 'p2', 'MaxPool (2) \u2014 16 Activation Maps'),
    'conv-details':         initConvDetails,
    'conv-full-digit':      initConvFullDigit,
    'conv-filters':         initConvFilters,
    'mlp-architecture':     initFullNetwork,
    'mnist-to-vector':      initMNISTToVector,
    'output-layer':         initOutputLayer,
    'perceptron':           initPerceptron,
    'training-network':     initTrainingNetwork,
  };

  const initialized = new Set();

  function maybeInit(id) {
    if (!SLIDES[id] || initialized.has(id)) return;
    const el = document.getElementById(id + '-canvas');
    if (!el) return;
    initialized.add(id);
    SLIDES[id](el);
  }

  // Re-run animation on revisit (clear initialized flag)
  function reinit(id) {
    if (!SLIDES[id]) return;
    const el = document.getElementById(id + '-canvas');
    if (!el) return;
    // Cancel any in-flight animation from previous run
    if (el._cancelAnim) { el._cancelAnim(); el._cancelAnim = null; }
    if (el._timers) { el._timers.forEach(clearTimeout); el._timers = []; }
    initialized.delete(id);
    maybeInit(id);
  }

  // By the time this script runs, DOMContentLoaded has already fired.
  // Register directly; Reveal.initialize() is async so 'ready' may not have
  // fired yet — but slidechanged always needs to be attached immediately.
  if (typeof Reveal !== 'undefined') {
    Reveal.on('slidechanged', (e) => {
      const id = e.currentSlide && e.currentSlide.id;
      if (id) reinit(id);
    });

    if (Reveal.isReady()) {
      // Reveal already finished initialising (unlikely but safe)
      const cur = Reveal.getCurrentSlide();
      if (cur) maybeInit(cur.id);
    } else {
      Reveal.on('ready', () => {
        const cur = Reveal.getCurrentSlide();
        if (cur) maybeInit(cur.id);
      });
    }
  }

})();
