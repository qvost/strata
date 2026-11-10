# STRATA Research Portal

Next.js App Router, React, TypeScript, Effect, react-markdown, KaTeX, and highlight.js via rehype-highlight.

## Run

```sh
npm install
npm run dev
```

Open http://localhost:3000. For production, run `npm run typecheck && npm run build && npm start` with Node.js 20.9 or later.

## Content and implementation

- `app/page.tsx`: editorial research portal, architecture, paper-sourced benchmark tables, prospective confirmation protocol, and all four negative results.
- `app/paper/page.tsx`: dynamically reads the complete `strata_research_paper.md` on the server. GFM tables, inline/display math, syntax highlighting, section anchors, and a separate illustrative kernel companion.
- `components/PaperActions.tsx`: copy citation metadata; PDF export through the browser print dialog (select Save as PDF).
- `lib/effect/simulator.ts`: Effect Ref beam control, scheduled 50 ms telemetry Stream, interruptible telemetry and animation fibers.
- `components/Field.tsx`: procedural 4×4 Bayer quantization fields, binary matrix, checkmark distance field, HNSW graph, and pointer-reactive 14 px grid with 0.93 decay.

Telemetry is an educational simulation, not a browser performance measurement. The requested 25,581 QPS / 96.2% L3 miss claims do not appear in the supplied paper. The evidence record labels them unverified rather than conflating them with the Cohere-1M results (1,415 QPS and 98.1% Recall@10). The requested ef sweep is a prospective reproduction protocol; no measurements are fabricated.

Google Fonts are loaded through CSS, with local fallbacks. Content and mathematics remain readable if external fonts are unavailable. Reduced motion disables mechanical text animations and freezes procedural time.

## Deployment and verification

The target is a Node.js Next.js server (`output: standalone`). Keep `strata_research_paper.md` in the runtime working directory. A private Sites record exists in `.openai/hosting.json`; it is unpublished. A Cloudflare deployment needs a compatible Next.js adapter or an explicit static-export strategy; the Node runtime cannot be uploaded as a Worker unchanged.

Production compilation and TypeScript checking pass. Next.js uses its supported TypeScript compiler API because the CLI subprocess returns empty configuration output in this environment. Browser visual QA has not been performed.

The interface includes a short logo reveal using `public/strata-logo.svg`, copied from the repository asset. Reduced-motion users bypass the reveal. Architecture and simulator surfaces share the paper-and-ink theme; pausing the simulator also pauses the graph animation.
