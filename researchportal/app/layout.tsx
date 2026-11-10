import PreloaderReveal from '@/components/PreloaderReveal';
import type { Metadata } from 'next';
import './globals.css';
import 'katex/dist/katex.min.css';
export const metadata: Metadata = { title: 'STRATA — Breaking the Memory Wall', icons: { icon: '/strata-logo.svg' }, description: 'Siddid Pandey’s research into two-stage quantized HNSW routing. Explore the architecture, evidence, and interactive simulator.' };
export default function RootLayout({ children }: {
    children: React.ReactNode;
}) { return <html lang="en"><body><PreloaderReveal />{children}</body></html>; }
