'use client';
import { useEffect, useRef } from 'react';
import { startAnimation } from '@/lib/effect/simulator';
const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
export default function Field({ mode = 'hero', stage = 'COARSE_BQ', ef = 80, running = true }: {
    mode?: 'hero' | 'bq' | 'bits' | 'fine' | 'graph';
    stage?: string;
    ef?: number;
    running?: boolean;
}) { const ref = useRef<HTMLCanvasElement>(null); useEffect(() => { const canvas = ref.current!; const ctx = canvas.getContext('2d')!; let w = 0, h = 0; const mouse = { x: -1000, y: -1000 }; let scrollProgress = 0; const onScroll = () => { const box = canvas.getBoundingClientRect(); scrollProgress = Math.max(0, Math.min(1, (innerHeight - box.top) / (innerHeight + box.height))); }; if (mode === 'graph') {
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
} let energy: Float32Array; const resize = () => { const box = canvas.getBoundingClientRect(); w = box.width; h = box.height; const dpr = Math.min(devicePixelRatio, 2); canvas.width = w * dpr; canvas.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); energy = new Float32Array(Math.ceil(w / 14) * Math.ceil(h / 14)); }; const observer = new ResizeObserver(resize); observer.observe(canvas); resize(); const move = (e: PointerEvent) => { const box = canvas.getBoundingClientRect(); mouse.x = e.clientX - box.left; mouse.y = e.clientY - box.top; }; canvas.addEventListener('pointermove', move); const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches; const render = (time: number) => { const t = reduced ? 1 : time * .00025; ctx.clearRect(0, 0, w, h); if (mode === 'hero' || mode === 'graph') {
    const nodes = Array.from({ length: mode === 'hero' ? 85 : 65 }, (_, i) => { const layer = i % 3; const a = i * 2.39996; const r = Math.sqrt((i + 1) / 85); return { x: w * .53 + Math.cos(a) * r * w * .43, y: h * .52 + Math.sin(a) * r * h * .4 - layer * 12 }; });
    ctx.lineWidth = .7;
    for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
            const b = nodes[j];
            const d = Math.hypot(a.x - b.x, a.y - b.y);
            if (d < w * .16) {
                ctx.strokeStyle = mode === 'hero' ? 'rgba(194,65,12,.2)' : 'rgba(10,10,10,.17)';
                ctx.beginPath();
                ctx.moveTo(a.x, a.y);
                ctx.lineTo(b.x, b.y);
                ctx.stroke();
            }
        }
        const active = (i + Math.floor(t * 12)) % 17 < 3;
        ctx.fillStyle = active ? '#ff6a2b' : mode === 'hero' ? '#b5aca0' : '#77746a';
        ctx.fillRect(a.x - 2, a.y - 2, active ? 6 : 3, active ? 6 : 3);
    }
    const cols = Math.ceil(w / 14);
    for (let y = 0; y < h; y += 14)
        for (let x = 0; x < w; x += 14) {
            const i = Math.floor(y / 14) * cols + Math.floor(x / 14);
            const d = Math.hypot(x - mouse.x, y - mouse.y);
            energy[i] = Math.max(energy[i] * .93, d < 70 ? (1 - d / 70) : 0);
            if (energy[i] > .02) {
                ctx.fillStyle = `rgba(255,106,43,${energy[i]})`;
                ctx.fillRect(x, y, 3 + energy[i] * 6, 3 + energy[i] * 6);
            }
        }
    if (mode === 'graph') {
        const progress = Math.min(.95, scrollProgress * .75 + (ef / 320) * .2);
        ctx.strokeStyle = '#c2410c';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(w * .53, h * .52, (1 - progress) * w * .35 + 15, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = '#c2410c';
        ctx.font = '12px monospace';
        ctx.fillText(`${stage} / BEAM ${ef}`, 18, h - 18);
    }
    return;
} for (let y = 0; y < h; y += 5)
    for (let x = 0; x < w; x += 5) {
        const nx = (x - w / 2) / w, ny = (y - h / 2) / h;
        let v = 0;
        if (mode === 'bq') {
            v = (Math.sin(nx * 18 + t * 3) + Math.cos(ny * 18 - t * 2) + Math.sin((nx + ny) * 15)) / 5 + .35;
        }
        else if (mode === 'fine') {
            const ax = -.25, ay = .02, bx = -.04, by = .23, cx = .3, cy = -.22;
            const dist = (x1: number, y1: number, x2: number, y2: number) => { const dx = x2 - x1, dy = y2 - y1; const q = Math.max(0, Math.min(1, ((nx - x1) * dx + (ny - y1) * dy) / (dx * dx + dy * dy))); return Math.hypot(nx - x1 - q * dx, ny - y1 - q * dy); };
            v = Math.max(0, 1 - Math.min(dist(ax, ay, bx, by), dist(bx, by, cx, cy)) * 17) * (0.7 + Math.sin(t * 2) * .2);
        }
        else {
            if (x % 10 || y % 15)
                continue;
            ctx.fillStyle = `rgba(10,10,10,${.18 + ((Math.sin(x * 13 + y * 7 + Math.floor(t * 5)) + 1) / 2) * .7})`;
            ctx.font = '12px monospace';
            ctx.fillText(Math.sin(x * 9 + y * 3 + Math.floor(t * 4)) > 0 ? '1' : '0', x, y);
            continue;
        }
        if (v > bayer[(Math.floor(x / 5) % 4) + 4 * (Math.floor(y / 5) % 4)] / 16) {
            ctx.fillStyle = '#c2410c';
            ctx.fillRect(x, y, 3, 3);
        }
    } }; render(performance.now()); const stop = running && !reduced ? startAnimation(render) : () => {};  return () => { stop(); observer.disconnect(); canvas.removeEventListener('pointermove', move); window.removeEventListener('scroll', onScroll); }; }, [mode, stage, ef, running]); return <canvas id={mode === 'hero' ? 'hero-kv' : undefined} ref={ref} className={`field field-${mode}`} aria-label={mode === 'hero' ? 'Interactive hierarchical proximity graph' : `${mode} procedural visualization`} role="img"/>; }
