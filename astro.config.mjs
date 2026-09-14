import mdx from "@astrojs/mdx";
import { rehypeHeadingIds, unified } from "@astrojs/markdown-remark";
import { defineConfig } from "astro/config";
import { fileURLToPath } from "node:url";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import rehypeEquationNumbers from "./tools/public/rehype-equation-numbers.mjs";
import rehypeHeadingPermalinks from "./tools/public/rehype-heading-permalinks.mjs";

import rehypeTableNumbers from "./tools/public/rehype-table-numbers.mjs";

import rehypeListingNumbers from "./tools/public/rehype-listing-numbers.mjs";
import rehypeFigureReferences, {
  figureReferenceDevPlugin,
} from "./tools/public/rehype-figure-references.mjs";

const projectRoot = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  site: "https://photontransport.com/",
  trailingSlash: "always",
  output: "static",
  integrations: [mdx()],
  vite: {
    define: {
      "import.meta.env.DPT_PROJECT_ROOT": JSON.stringify(projectRoot),
    },
    plugins: [figureReferenceDevPlugin({ root: projectRoot })],
  },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeKatex,
        rehypeEquationNumbers,
        rehypeTableNumbers,
        rehypeListingNumbers,
        [rehypeFigureReferences, { root: projectRoot }],
        // Use Astro's own slugs before appending text-free permalink icons.
        rehypeHeadingIds,
        rehypeHeadingPermalinks,
      ],
    }),
    shikiConfig: {
      theme: "github-light",
      wrap: true,
    },
  },
});
