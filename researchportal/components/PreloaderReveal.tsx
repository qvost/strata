'use client';

import { useEffect, useState } from 'react';
import styles from './PreloaderReveal.module.css';

/** A short brand reveal; CSS also dismisses it if hydration is delayed. */
export default function PreloaderReveal() {
  const [visible, setVisible] = useState(true);
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setVisible(false);
      return;
    }
    const timer = window.setTimeout(() => setVisible(false), 1450);
    return () => window.clearTimeout(timer);
  }, []);
  if (!visible) return null;
  return <div className={styles.reveal} aria-hidden="true">
    <div className={styles.identity}>
      <img src="/strata-logo.svg" width="88" height="88" alt="" loading="eager" />
      <strong>STRATA</strong>
      <span>Two-stage vector traversal</span>
    </div>
    <div className={styles.rule} />
  </div>;
}
