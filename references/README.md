# Reference store

`library.json` is the canonical CSL-JSON bibliography. It contains the original
`siddon1985` path-calculation paper and the 20 sources used by Appendix A.

Chapter MDX refers to stable keys, for example
`<Citation keys={["siddon1985"]} />`. Formatted references are never canonical.
Replace `styles/default.csl` to change the rendering style without editing
chapter content. `overrides.yml` is JSON-compatible YAML and is reserved for
documented corrections to imported DOI, arXiv, or PMID metadata.

## Metadata conventions

- Keep complete author or editor lists, institutional authors as `literal`
  names, and article identifiers in `page` when a journal uses them instead of
  page ranges.
- Verify journal metadata against the publisher's Crossref record. Check books,
  datasets, standards and guidance against their official publication records.
  A valid identifier establishes identity, not support for a particular claim.
- Use the publication date, not a later website migration or modification date.
  Undated guidance uses an empty `issued.date-parts` array and renders as `n.d.`;
  `accessed` records when it was consulted.
- Record the edition of a changing standard. The DICOM current-edition URL
  resolved to PS3.3 2026c when checked on 7 September 2026.
- Distinguish the May 1995 NISTIR 5632 report from its subsequently updated online
  tables, and XCOM's November 2010 data update from changes to its web page.

The Appendix A metadata was checked on 7 September 2026. Canonical titles and
complete author lists replace abbreviated input citations; journal spelling is
preserved, including “computerised” in Alvarez and Macovski's 1976 title. The
IAEA handbook lists technical editors, not joint authors. Otake's author
manuscript resolves the imported name-parsing ambiguity for J. Webster Stayman.
