'use client';
import { useState } from 'react';
const bibtex = '@techreport{pandey_strata,\n  title = {STRATA: Breaking the Memory-Bandwidth Wall in HNSW Graph Traversal with Two-Stage Quantized Routing},\n  author = {Pandey, Siddid},\n  url = {https://github.com/qvost/strata}\n}';
export default function PaperActions() { const [status, setStatus] = useState('Copy BibTeX'); async function copy() { try {
    await navigator.clipboard.writeText(bibtex);
    setStatus('BibTeX copied');
}
catch {
    setStatus('Clipboard unavailable');
} } return <div className="paper-actions"><button onClick={() => window.print()}>Download PDF <span>↗</span></button><button onClick={copy}>{status}</button><span className="sr-only" role="status">{status}</span></div>; }
