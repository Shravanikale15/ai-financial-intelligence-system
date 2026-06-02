// ===== SVG CHART ENGINE =====
const Charts = {
  // Smooth line chart with cubic bezier
  lineChart(containerId, data, options = {}) {
    const c = document.getElementById(containerId);
    if (!c) return;
    c.innerHTML = '';

    const W = options.width || c.clientWidth || 500;
    const H = options.height || 260;
    const pad = { top: 20, right: 20, bottom: 30, left: 50 };
    const w = W - pad.left - pad.right;
    const h = H - pad.top - pad.bottom;

    const lines = options.lines || [{ key: 'profit', color: '#3B82F6', dash: false }, { key: 'loss', color: '#18181B', dash: true }];
    const labels = data.map(d => d.month || d.label);

    // Find min/max
    let allVals = [];
    lines.forEach(l => data.forEach(d => allVals.push(d[l.key] || 0)));
    const maxVal = Math.max(...allVals) * 1.15;
    const minVal = Math.min(0, Math.min(...allVals));

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', H);
    svg.style.overflow = 'visible';

    // Grid lines
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (h / 4) * i;
      const val = Math.round(maxVal - (maxVal - minVal) * (i / 4));
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', pad.left); line.setAttribute('x2', W - pad.right);
      line.setAttribute('y1', y); line.setAttribute('y2', y);
      line.setAttribute('stroke', '#F0F0F0'); line.setAttribute('stroke-width', '1');
      svg.appendChild(line);

      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', pad.left - 8); text.setAttribute('y', y + 4);
      text.setAttribute('text-anchor', 'end');
      text.setAttribute('fill', '#9CA3AF'); text.setAttribute('font-size', '10');
      text.setAttribute('font-family', 'Outfit');
      text.textContent = val >= 1000 ? Math.round(val / 1000) + 'k' : val;
      svg.appendChild(text);
    }

    // X labels
    labels.forEach((label, i) => {
      const x = pad.left + (w / (labels.length - 1)) * i;
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', x); text.setAttribute('y', H - 6);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', '#9CA3AF'); text.setAttribute('font-size', '10');
      text.setAttribute('font-family', 'Outfit');
      text.textContent = label;
      svg.appendChild(text);
    });

    // Draw lines
    lines.forEach(lineOpt => {
      const points = data.map((d, i) => ({
        x: pad.left + (w / (data.length - 1)) * i,
        y: pad.top + h - ((d[lineOpt.key] - minVal) / (maxVal - minVal)) * h
      }));

      // Build smooth path
      let pathD = `M ${points[0].x} ${points[0].y}`;
      for (let i = 1; i < points.length; i++) {
        const prev = points[i - 1];
        const curr = points[i];
        const cpx1 = prev.x + (curr.x - prev.x) * 0.4;
        const cpx2 = curr.x - (curr.x - prev.x) * 0.4;
        pathD += ` C ${cpx1} ${prev.y} ${cpx2} ${curr.y} ${curr.x} ${curr.y}`;
      }

      // Gradient fill
      if (!lineOpt.dash) {
        const gradId = `grad-${lineOpt.key}-${containerId}`;
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const grad = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        grad.setAttribute('id', gradId);
        grad.setAttribute('x1', '0'); grad.setAttribute('y1', '0');
        grad.setAttribute('x2', '0'); grad.setAttribute('y2', '1');

        const s1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        s1.setAttribute('offset', '0%'); s1.setAttribute('stop-color', lineOpt.color); s1.setAttribute('stop-opacity', '0.15');
        const s2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
        s2.setAttribute('offset', '100%'); s2.setAttribute('stop-color', lineOpt.color); s2.setAttribute('stop-opacity', '0');
        grad.appendChild(s1); grad.appendChild(s2);
        defs.appendChild(grad);
        svg.appendChild(defs);

        const areaD = pathD + ` L ${points[points.length - 1].x} ${pad.top + h} L ${points[0].x} ${pad.top + h} Z`;
        const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        area.setAttribute('d', areaD);
        area.setAttribute('fill', `url(#${gradId})`);
        svg.appendChild(area);
      }

      // Line path
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', pathD);
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', lineOpt.color);
      path.setAttribute('stroke-width', '2.5');
      if (lineOpt.dash) path.setAttribute('stroke-dasharray', '6 3');

      // Animate line draw
      const len = path.getTotalLength ? 1000 : 0;
      if (len) {
        path.style.strokeDasharray = '1000';
        path.style.strokeDashoffset = '1000';
        path.style.transition = 'stroke-dashoffset 1.5s ease';
        setTimeout(() => { path.style.strokeDashoffset = '0'; }, 100);
      }
      svg.appendChild(path);

      // Dots
      points.forEach((p, i) => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', p.x); circle.setAttribute('cy', p.y);
        circle.setAttribute('r', '5');
        circle.setAttribute('fill', lineOpt.color);
        circle.setAttribute('stroke', 'white');
        circle.setAttribute('stroke-width', '2');
        circle.style.opacity = '0';
        circle.style.transition = `opacity 0.3s ease ${i * 0.1}s`;
        setTimeout(() => { circle.style.opacity = '1'; }, 200);

        // Tooltip
        circle.addEventListener('mouseenter', e => {
          Charts._showTooltip(e, `${labels[i]}: ₹${(data[i][lineOpt.key] || 0).toLocaleString()}`);
        });
        circle.addEventListener('mouseleave', () => Charts._hideTooltip());
        circle.style.cursor = 'pointer';
        svg.appendChild(circle);
      });
    });

    c.appendChild(svg);
  },

  // Donut chart
  donutChart(containerId, segments, options = {}) {
    const c = document.getElementById(containerId);
    if (!c) return;
    c.innerHTML = '';

    const size = options.size || 200;
    const thickness = options.thickness || 30;
    const cx = size / 2, cy = size / 2, r = (size - thickness) / 2;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
    svg.setAttribute('width', size); svg.setAttribute('height', size);

    // Background circle
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    bg.setAttribute('cx', cx); bg.setAttribute('cy', cy); bg.setAttribute('r', r);
    bg.setAttribute('fill', 'none'); bg.setAttribute('stroke', '#F3F4F6');
    bg.setAttribute('stroke-width', thickness);
    svg.appendChild(bg);

    let cumPct = 0;
    const circumference = 2 * Math.PI * r;

    segments.forEach((seg, i) => {
      const segLen = (seg.pct / 100) * circumference;
      const offset = cumPct * circumference / 100;

      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', cx); circle.setAttribute('cy', cy); circle.setAttribute('r', r);
      circle.setAttribute('fill', 'none');
      circle.setAttribute('stroke', seg.color);
      circle.setAttribute('stroke-width', thickness);
      circle.setAttribute('stroke-dasharray', `${segLen} ${circumference - segLen}`);
      circle.setAttribute('stroke-dashoffset', -offset);
      circle.setAttribute('transform', `rotate(-90 ${cx} ${cy})`);
      circle.style.transition = `stroke-dasharray 1s ease ${i * 0.2}s`;

      circle.addEventListener('mouseenter', e => {
        Charts._showTooltip(e, `${seg.name}: ${seg.pct}%`);
      });
      circle.addEventListener('mouseleave', () => Charts._hideTooltip());
      circle.style.cursor = 'pointer';

      svg.appendChild(circle);
      cumPct += seg.pct;
    });

    // Center text
    if (options.centerText) {
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('x', cx); t.setAttribute('y', cy - 4);
      t.setAttribute('text-anchor', 'middle'); t.setAttribute('font-size', '24');
      t.setAttribute('font-weight', '700'); t.setAttribute('fill', '#111827');
      t.setAttribute('font-family', 'Outfit');
      t.textContent = options.centerText;
      svg.appendChild(t);

      if (options.centerSub) {
        const s = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        s.setAttribute('x', cx); s.setAttribute('y', cy + 16);
        s.setAttribute('text-anchor', 'middle'); s.setAttribute('font-size', '11');
        s.setAttribute('fill', '#9CA3AF'); s.setAttribute('font-family', 'Outfit');
        s.textContent = options.centerSub;
        svg.appendChild(s);
      }
    }

    c.appendChild(svg);
  },

  // Radial progress (health score)
  radialProgress(containerId, value, max = 100) {
    const c = document.getElementById(containerId);
    if (!c) return;
    c.innerHTML = '';

    const size = 130, thickness = 10;
    const cx = size / 2, cy = size / 2, r = (size - thickness * 2) / 2;
    const circumference = 2 * Math.PI * r;
    const progress = (value / max) * circumference;

    let color = '#10B981';
    if (value < 40) color = '#EF4444';
    else if (value < 70) color = '#F59E0B';

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
    svg.setAttribute('width', size); svg.setAttribute('height', size);

    // BG
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    bg.setAttribute('cx', cx); bg.setAttribute('cy', cy); bg.setAttribute('r', r);
    bg.setAttribute('fill', 'none'); bg.setAttribute('stroke', '#F3F4F6');
    bg.setAttribute('stroke-width', thickness);
    svg.appendChild(bg);

    // Progress
    const prog = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    prog.setAttribute('cx', cx); prog.setAttribute('cy', cy); prog.setAttribute('r', r);
    prog.setAttribute('fill', 'none'); prog.setAttribute('stroke', color);
    prog.setAttribute('stroke-width', thickness);
    prog.setAttribute('stroke-linecap', 'round');
    prog.setAttribute('stroke-dasharray', `${progress} ${circumference - progress}`);
    prog.setAttribute('transform', `rotate(-90 ${cx} ${cy})`);
    prog.style.transition = 'stroke-dasharray 1.5s ease';
    svg.appendChild(prog);

    // Value text
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', cx); t.setAttribute('y', cy + 2);
    t.setAttribute('text-anchor', 'middle'); t.setAttribute('font-size', '30');
    t.setAttribute('font-weight', '700'); t.setAttribute('fill', '#111827');
    t.setAttribute('font-family', 'Outfit');
    t.textContent = value;
    svg.appendChild(t);

    const l = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    l.setAttribute('x', cx); l.setAttribute('y', cy + 18);
    l.setAttribute('text-anchor', 'middle'); l.setAttribute('font-size', '10');
    l.setAttribute('fill', '#9CA3AF'); l.setAttribute('font-family', 'Outfit');
    l.textContent = 'SCORE';
    svg.appendChild(l);

    c.appendChild(svg);
  },

  // Sparkline
  sparkline(containerId, values, color = '#10B981') {
    const c = document.getElementById(containerId);
    if (!c) return;
    c.innerHTML = '';

    const W = 100, H = 36;
    const max = Math.max(...values);
    const min = Math.min(...values);

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('width', '100%'); svg.setAttribute('height', H);

    const points = values.map((v, i) => ({
      x: (i / (values.length - 1)) * W,
      y: H - 4 - ((v - min) / (max - min || 1)) * (H - 8)
    }));

    let d = `M ${points[0].x} ${points[0].y}`;
    for (let i = 1; i < points.length; i++) {
      const p = points[i - 1], c2 = points[i];
      d += ` C ${p.x + (c2.x - p.x) * 0.4} ${p.y} ${c2.x - (c2.x - p.x) * 0.4} ${c2.y} ${c2.x} ${c2.y}`;
    }

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', d); path.setAttribute('fill', 'none');
    path.setAttribute('stroke', color); path.setAttribute('stroke-width', '2');
    path.setAttribute('stroke-linecap', 'round');
    svg.appendChild(path);

    c.appendChild(svg);
  },

  // Tooltip helper
  _tooltip: null,
  _showTooltip(e, text) {
    if (!Charts._tooltip) {
      Charts._tooltip = document.createElement('div');
      Charts._tooltip.style.cssText = 'position:fixed;background:#18181B;color:white;padding:6px 12px;border-radius:8px;font-size:12px;font-family:Outfit;pointer-events:none;z-index:9999;white-space:nowrap;box-shadow:0 4px 12px rgba(0,0,0,0.15);';
      document.body.appendChild(Charts._tooltip);
    }
    Charts._tooltip.textContent = text;
    Charts._tooltip.style.display = 'block';
    Charts._tooltip.style.left = (e.clientX + 12) + 'px';
    Charts._tooltip.style.top = (e.clientY - 30) + 'px';
  },
  _hideTooltip() {
    if (Charts._tooltip) Charts._tooltip.style.display = 'none';
  }
};
