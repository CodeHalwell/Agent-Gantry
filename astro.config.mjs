import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

export default defineConfig({
  integrations: [react()],
  site: 'https://codehalwell.github.io',
  base: '/Agent-Gantry',
  output: 'static',
});
