'use client';

import Link from 'next/link';
import { useEffect, useRef, type PointerEvent } from 'react';
import { startAnimation } from '@/lib/effect/simulator';
import { stepTags, type TagBody } from '@/lib/footerPhysics';
import styles from './PhysicsTagFooter.module.css';

const contacts = [
  { name: 'GitHub', detail: '@qvost', href: 'https://github.com/qvost', icon: 'github' },
  { name: 'LinkedIn', detail: '/in/qvost', href: 'https://www.linkedin.com/in/qvost', icon: 'linkedin' },
  { name: 'Email', detail: 'qvostwon@gmail.com', href: 'mailto:qvostwon@gmail.com', icon: 'email' },
  { name: 'Phone', detail: '9801004118', href: 'tel:9801004118', icon: 'phone' },
] as const;

function ContactIcon({ kind }: { kind: typeof contacts[number]['icon'] }) {
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {kind === 'github' && <path d="M9 19c-4.5 1.5-4.5-2.5-6-3m12 6v-3.9a3.4 3.4 0 0 0-1-2.6c3.3-.4 6.8-1.6 6.8-7.3A5.7 5.7 0 0 0 19.3 4a5.3 5.3 0 0 0-.1-4S18 0 15 1.6a14.7 14.7 0 0 0-8 0C4 0 2.8 0 2.8 0a5.3 5.3 0 0 0-.1 4A5.7 5.7 0 0 0 1.2 8c0 5.7 3.5 6.9 6.8 7.3a3.4 3.4 0 0 0-1 2.6V22" transform="translate(2 2) scale(.85)" />}
    {kind === 'linkedin' && <><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M7 10v7m0-10v.1M11 17v-7m0 3a3 3 0 0 1 6 0v4" /></>}
    {kind === 'email' && <><rect x="3" y="5" width="18" height="14" rx="2" /><path d="m3 6 9 7 9-7" /></>}
    {kind === 'phone' && <path d="m7 3 3 5-3 2a15 15 0 0 0 7 7l2-3 5 3c-1 5-5 5-9 2C6 16 1 8 3 5c1-2 2-2 4-2Z" />}
  </svg>;
}

export default function PhysicsTagFooter() {
  const stage = useRef<HTMLDivElement>(null);
  const tags = useRef<(HTMLAnchorElement | null)[]>([]);
  const bodies = useRef<TagBody[]>([]);
  const drag = useRef<{ index: number; pointer: number; offsetX: number; offsetY: number; startX: number; startY: number; lastX: number; lastY: number; time: number } | null>(null);
  const suppressClick = useRef(false);

  useEffect(() => {
    const container = stage.current!;
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    let width = 0;
    let height = 0;
    let visible = false;
    let previous = 0;
    let stop: (() => void) | undefined;
    const paint = () => bodies.current.forEach((body, i) => {
      const tag = tags.current[i];
      if (tag) tag.style.transform = `translate3d(${body.x}px, ${body.y}px, 0)`;
    });
    const measure = () => {
      width = container.clientWidth;
      height = container.clientHeight;
      if (media.matches) return;
      bodies.current = tags.current.map((tag, i) => {
        const old = bodies.current[i];
        const w = tag!.offsetWidth;
        const h = tag!.offsetHeight;
        return { x: Math.max(0, Math.min(width - w, old?.x ?? (width - w) * [0.04, .62, .2, .85][i])), y: Math.min(height - h, old?.y ?? 15 + i * 57), vx: old?.vx ?? [40, -35, 28, -45][i], vy: old?.vy ?? 0, width: w, height: h, held: false };
      });
      drag.current = null;
      paint();
    };
    const sync = () => {
      stop?.();
      stop = undefined;
      previous = 0;
      if (!visible || document.hidden || media.matches) return;
      stop = startAnimation(time => {
        const dt = previous ? Math.min((time - previous) / 1000, .04) : 1 / 60;
        previous = time;
        // Substeps keep dragged tags and fast throws from crossing thin bodies.
        for (let i = 0; i < 3; i++) stepTags(bodies.current, width, height, dt / 3);
        paint();
      });
    };
    const onMotionChange = () => { measure(); sync(); };
    const resize = new ResizeObserver(measure);
    resize.observe(container);
    const intersection = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); });
    intersection.observe(container);
    document.addEventListener('visibilitychange', sync);
    media.addEventListener('change', onMotionChange);
    measure();
    return () => {
      stop?.();
      resize.disconnect();
      intersection.disconnect();
      document.removeEventListener('visibilitychange', sync);
      media.removeEventListener('change', onMotionChange);
      bodies.current = [];
    };
  }, []);

  function pickUp(event: PointerEvent<HTMLAnchorElement>, index: number) {
    if (event.button !== 0 || drag.current || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const body = bodies.current[index];
    if (!body) return;
    const box = stage.current!.getBoundingClientRect();
    suppressClick.current = false;
    body.held = true;
    body.vx = body.vy = 0;
    drag.current = { index, pointer: event.pointerId, offsetX: event.clientX - box.left - body.x, offsetY: event.clientY - box.top - body.y, startX: event.clientX, startY: event.clientY, lastX: event.clientX, lastY: event.clientY, time: performance.now() };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.dataset.held = 'true';
  }

  function move(event: PointerEvent<HTMLAnchorElement>) {
    const held = drag.current;
    if (!held || event.pointerId !== held.pointer) return;
    const body = bodies.current[held.index];
    const box = stage.current!.getBoundingClientRect();
    const now = performance.now();
    const elapsed = Math.max(.008, (now - held.time) / 1000);
    body.x = Math.max(0, Math.min(box.width - body.width, event.clientX - box.left - held.offsetX));
    body.y = Math.max(0, Math.min(box.height - body.height, event.clientY - box.top - held.offsetY));
    body.vx = Math.max(-900, Math.min(900, (event.clientX - held.lastX) / elapsed));
    body.vy = Math.max(-900, Math.min(900, (event.clientY - held.lastY) / elapsed));
    if (Math.hypot(event.clientX - held.startX, event.clientY - held.startY) > 6) suppressClick.current = true;
    held.lastX = event.clientX;
    held.lastY = event.clientY;
    held.time = now;
    event.currentTarget.style.transform = `translate3d(${body.x}px, ${body.y}px, 0)`;
  }

  function release(event: PointerEvent<HTMLAnchorElement>) {
    const held = drag.current;
    if (!held || held.pointer !== event.pointerId) return;
    const body = bodies.current[held.index];
    body.held = false;
    if (performance.now() - held.time > 100) body.vx = body.vy = 0;
    drag.current = null;
    delete event.currentTarget.dataset.held;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }

  return <footer className={styles.footer} aria-labelledby="footer-title">
    <div className="wrap">
      <div className={styles.heading}>
        <div><h2 id="footer-title">Good systems start<br />with a conversation<span className="orange">.</span></h2><p>Research, open source, or a difficult systems problem.<br />Let’s compare notes.</p></div>
        <div className={styles.identity}><strong>Siddid Pandey</strong><span>Kathmandu, Nepal</span></div>
      </div>
      <div className={styles.stage} ref={stage} aria-label="Contact links. Drag and release the tags, or click to connect.">
        <span className={styles.hint} aria-hidden="true">A little gravity. A direct connection. <span>↙ Grab a tag</span></span>
        <span className={styles.watermark} aria-hidden="true">LET’S TALK</span>
        {contacts.map((contact, index) => <a key={contact.name} ref={node => { tags.current[index] = node; }} href={contact.href} className={styles.tag} data-kind={contact.icon} target={contact.href.startsWith('https:') ? '_blank' : undefined} rel={contact.href.startsWith('https:') ? 'noopener noreferrer' : undefined} aria-label={`${contact.name}: ${contact.detail}`} onPointerDown={event => pickUp(event, index)} onPointerMove={move} onPointerUp={release} onPointerCancel={release} onLostPointerCapture={release} onDragStart={event => event.preventDefault()} onClick={event => { if (suppressClick.current && event.detail !== 0) event.preventDefault(); suppressClick.current = false; }}>
          <span className={styles.icon}><ContactIcon kind={contact.icon} /></span><span className={styles.contact}><strong>{contact.name}</strong><span>{contact.detail}</span></span><span className={styles.arrow} aria-hidden="true">↗</span>
        </a>)}
      </div>
      <div className={styles.bottom}><Link href="/" className="brand">STRATA</Link><span>Independent research by Siddid Pandey</span><a href="#">Back to top ↑</a></div>
    </div>
  </footer>;
}
