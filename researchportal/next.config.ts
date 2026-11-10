import type { NextConfig } from 'next';

const config: NextConfig = {
  output: 'standalone',
  // The CLI subprocess returns empty config output in this environment.
  // Use the supported compiler API so production builds retain type checking.
  experimental: { useTypeScriptCli: false },
  outputFileTracingIncludes: { '/*': ['./strata_research_paper.md'] },
};

export default config;
