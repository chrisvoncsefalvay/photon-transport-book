# Website deployment

The book is a static Astro site at <https://photontransport.com/>. Deploy only
the generated, allowlisted public source tree. Its canonical authoring checkout
contains private material and is rejected by the hosting preflight.

## Build a website snapshot

The public tree must include `public/generated/source-regions.json` generated
from the exact committed author sources used for its export. Its `source_commit`
is provenance; that commit need not exist in this repository. The deployment
build creates same-site source pages from the included files, preserving line
numbers and source hashes. It never links an author commit into a different
repository's Git history.

Generate this manifest after general source-generation checks and immediately
before the guarded build. The ordinary development `build` and `check` scripts
can replace its commit with the checkout's current Git HEAD; use `build:vercel`
for the deployment build so the exported author identity is retained.

Use Node.js 24 and the pinned package manager:

```sh
node tools/public/check-deployment.mjs --preinstall
corepack pnpm install --frozen-lockfile
corepack pnpm run build:vercel
```

The build validates the public inputs, source references, recorded artefacts and
links before producing `dist/`. It then checks canonical page URLs, source-line
links, snapshot metadata, forbidden output paths and the 25 MiB per-file limit.
`website-snapshot.json` records the author commit, canonical URL, unchanged
citation version and generated source-manifest hash. Vercel's public Git commit
is recorded separately when the deployment provides a verified repository
identity.

The website publishes the edition identified by `CITATION.cff`. Version 1.0.0,
released on 14 September 2026, is citable at <https://photontransport.com/>
without a DOI. Deployment preserves the authored version and release date.
Explicit prerelease versions, such as `0.0.0-bootstrap` or `1.1.0-rc.1`, retain
draft citations. Registering an archived edition with Zenodo remains a separate
process with its own reserved-DOI and frozen-citation checks.

## Vercel

The checked-in configuration selects Astro, Node 24, the guarded install/build
commands and `dist/`. It uses trailing slashes, redirects
`www.photontransport.com` to the canonical domain, caches fingerprinted Astro
assets and revalidates mutable generated files. Source-browser pages remain
available through code links and carry `X-Robots-Tag: noindex`.

Set `ENABLE_EXPERIMENTAL_COREPACK=1` in the project's build environments so
Vercel honours the exact `packageManager` version through its documented
[Corepack support](https://vercel.com/docs/builds/configure-a-build#corepack).

Run the CLI from the generated public tree, after selecting the intended Vercel
team and project:

```sh
vercel link
vercel deploy
```

Inspect the returned deployment and verify its chapters, source links and
`website-snapshot.json`. For the approved production publication, use
`vercel deploy --prod`, or promote a verified deployment with
`vercel promote <deployment-url>`.

The CLI path does not require Git integration. Automatic Git deployments remain
disabled in `vercel.json`; connecting Git alone does not change that policy.
When Git integration is enabled, it must identify
`chrisvoncsefalvay/photon-transport-book`. Do not put `DPT_LOCAL_SOURCE_LINKS`,
`DPT_SOURCE_COMMIT` or `DPT_SOURCE_LINK_REF` in hosting settings: the exported
source manifest supplies the build identity.

Vercel's deployment history retains previous static outputs. Roll back to a
verified deployment with `vercel rollback <deployment-url>` if necessary.

See the official [Astro deployment guide](https://docs.astro.build/en/guides/deploy/vercel/),
[Vercel configuration reference](https://vercel.com/docs/project-configuration/vercel-json)
and [Vercel CLI guide](https://vercel.com/docs/cli).
