# Terminology registries

These registries collect the vocabulary, notation and coordinate conventions of
Chapters 1–10 and Appendix A. They describe the current manuscript; they do not
claim that the scientific operators have been implemented or validated. The current
inventory contains 178 terms, 250 symbols and six frames or direction bases.

- `terms.yml` defines concepts, accepted aliases and distinctions that matter when
  interpreting the forward model or its derivatives. Entries are ordered by ID.
- `symbols.yml` records LaTeX, spoken forms, meanings, units and the section that
  establishes each notation convention.
- `frames.yml` records physical coordinate frames and explicitly identified local
  bases. Dimensionless voxel indices belong to the symbol registry: anisotropic
  spacing makes the physical-to-grid map affine, not a rigid frame transform.

## Provenance and scope

The current definitions are editorial summaries of the manuscript. Every record
links to its supporting book section: terms use `notes`; symbols and frames use
`introduced_in`, with qualifications in `notes`. Here `introduced_in` identifies
where the recorded convention is established, even if the notation is mentioned
informally earlier in the book. The chapter bibliography provides the surrounding
literature context.

`source_keys` is reserved for external references supporting a particular record.
An empty list means that this pass used the linked manuscript as its source; it
is not a claim of independently checked external support. When adding an external
source, check its support for the definition and use an existing key from
`references/library.json`. Do not copy every citation in a chapter into a record.

Aliases are alternative names for the same scoped concept, not loosely related
quantities. `avoid` is a list of misleading substitute labels for that entry; an
empty list imposes no additional wording prohibition. Qualifications in `notes`
remain part of the definition. These files are reference registries, not a claim
that every temporary index or intermediate expression has a global symbol.

## Conventions that must remain explicit

Physical coordinates and translations use millimetres, angles use radians, and
photon energies use keV unless an entry states otherwise. Attenuation coefficients
and integration distances must use reciprocal units. Dimensionless quantities
use `1`; a null unit means that a single physical unit does not
apply to that record.

Notation is scoped by meaning and section. A repeated LaTeX expression need not
have the same meaning in every chapter. In particular, an overbar can denote an
expected measurement or a reverse-mode cotangent; density and translational pose
coordinates both use rho. Retain distinct IDs for these uses. A detector score
is also different from a likelihood score.

`G_AB` maps column-vector coordinates from frame B to frame A. Local pose
coordinates list translation before rotation. Left increments act in world
coordinates and right increments in current object coordinates. Detector indices
are column then row, stored as `[j,i]`; volume samples use `[k,j,i]`. Patient LPS
orientation and displayed image orientation are separate choices.

Keep transmission, expected counts, detector signal and processed display values
separate. Likewise, distinguish acquisition noise from Monte Carlo uncertainty,
and physical exposure from the number of simulated histories. These distinctions
are needed to interpret derivatives and validation results correctly.

## Editing and validation

Keep stable, descriptive, lowercase hyphenated IDs. Preserve the JSON-compatible
YAML flow format used by the existing reader. Schemas live in `schemas/`; records
must satisfy those schemas and IDs must be unique within each registry.

Use explicit `$...$` inline LaTeX for mathematical variables and expressions in
`meaning`, `definition`, `notes`, `units` and frame `axes` descriptions. In these
JSON-compatible strings, escape LaTeX backslashes: `"$\\exp(-L_p(E))$"`, `"$T_p$"`
and `"$\\mathrm{mm}^{-1}$"` express a function, a subscripted variable and a unit
power. Keep ordinary prose, code, section references and URLs outside math
delimiters. Registry IDs, labels, spoken forms and aliases remain literal text;
the dedicated `latex` field remains an expression without dollar delimiters.

After editing, run from the repository root:

```sh
node tools/public/validate-registries.mjs
node tools/public/validate-all.mjs
```

The registry validator checks schemas, IDs and external citation-key existence.
It does not assess scientific claim support, unit correctness or book-link
fragments. Check those against the manuscript and built pages when definitions
change. The terminology directory is already included in the public allowlist;
private planning and review records belong outside it.

## Reader reference

The `/notation/` page renders symbols, terms and frames directly from these
registries, with units, qualifications and links to the defining sections. Its
search filters preserve stable entry links and the complete tables remain
available without JavaScript.

Chapter definition sidenotes use the same canonical meanings for terms and
registry symbols. An equation's authored `EquationNote` takes precedence for that
equation. Reused notation shows its recorded meanings with their source sections;
`introduced_in` is not treated as a claim that one meaning applies everywhere.
The typed visual presentation does not delay the accessible definition text.
