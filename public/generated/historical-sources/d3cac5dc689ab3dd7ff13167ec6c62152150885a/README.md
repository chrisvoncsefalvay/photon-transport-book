# Historical figure producer sources

These 79 files are exact source, configuration and dependency-lock bytes for
the transmission-contract and introduction-pose-sensitivity figures recorded
on 13 September 2026. They were recovered from repository commit
`d3cac5dc689ab3dd7ff13167ec6c62152150885a`. Every file first matched both the
SHA-256 in its original, unchanged run record and the Git blob at that commit.

`provenance.json` maps each original repository path to its archive path and
records the SHA-256, Git blob, consuming figures and unchanged output hashes.
Both figure manifests now validate these historical paths. Their original
`run.json` files, images, numerical arrays and validation results are unchanged.
No result was recomputed, and no old producer hash was replaced with a digest
of today's implementation.

The directory preserves the original relative layout. Its code is an archive,
not the current canonical Python package; do not add it to the current package
path. The live `python/dpt` tree remains the maintained implementation. Replaying
the historical introduction experiment still requires its separately governed
inputs; neither medical data nor third-party dependencies are included here.

Repository code and configurations retain the Apache-2.0 licence in
`LICENSE-CODE` at the repository root. Original authorship and any notices in
the source bytes are retained. This archive documentation follows the book's
CC-BY-NC-4.0 content licence.
