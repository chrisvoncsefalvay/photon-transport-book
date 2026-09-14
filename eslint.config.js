import eslintPluginAstro from "eslint-plugin-astro";
import tseslint from "typescript-eslint";

export default [
  {
    ignores: [
      ".astro/**",
      ".beagle/**",
      ".release/**",
      ".release-staging/**",
      ".venv/**",
      "dist/**",
      "node_modules/**",
      "public/generated/**",
    ],
  },
  ...tseslint.configs.recommended,
  ...eslintPluginAstro.configs["flat/recommended"],
];
