# Figure numbering and references

Declare each complete figure directly in its chapter MDX with a literal, stable
`id`. The shared registry assigns the displayed chapter number from reading
order; do not pass a `number` prop.

```mdx
[Figure](#figure-selected-ray) follows the ray through the slab.

<SelectedRayFigure id="figure-selected-ray" />
```

Use the same link syntax across chapters:
`[Figure](/chapters/transmission/#figure-selected-ray)`. The compiler fills the
number and preserves surrounding linked prose. Descriptive links without the
word “Figure” keep their wording. Captions and references appear in static HTML;
development also refreshes references when the target chapter changes.

Existing IDs such as `figure-4-1` are historical anchors. They remain unchanged
even if the current display number is 4.2. A figure's data or drawing selector
must depend on its stable ID or an explicit kind, never on its display number.

A wrapper passes its ID to `FigurePlate`. Register new complete-figure wrapper
names in `tools/public/lib/figure-registry.mjs`; inner charts and other surface
components are not independently numbered. The registry follows aliased default
imports. It rejects duplicate IDs, missing/dynamic IDs, manual numbers, nested
complete figures and runtime-generated figure declarations. An unresolved figure
reference is a build error.

The same registry supplies captions, Markdown reference rewriting and development
invalidation. The Astro configuration passes the project root explicitly, so
numbering does not depend on the shell's working directory.

`IntroductionStudy` follows the displayed CT capstone from its native coronal
CT plane through one simulated detector image to the recorded recovered bone
surface. The CT raster retains its physical pixel aspect; its HU window differs
from the radiograph's log window. The white model uses paper and the same mesh,
threshold and camera as the Chapter 12 static display. Geometry was fixed during
this material fit; the separate registration example is not an intermediate
step. The historical `figure-maisi-radiographs` anchor remains stable.

`ReconstructionRadiographs` uses separately curated recorded detector panels.
Each channel shares its fixed observation/prediction window; residual signs and
units remain explicit. The panels use paper and ink design tokens, retaining
the original radiograph pixels and palettes. A sliding radio selector in the
top bar selects discrete recorded channels with native keyboard controls. Without
JavaScript and in print, every channel remains available. CT withheld views and
acquired fitting frames retain their different evidential roles.

`ReconstructionFigure` uses the verified `generated/reconstruction/study.json`
contract. Its first render is a static plate in the book's figure system.
The paired CT view loads recorded fields and display meshes when visible;
the acquired diagnostic uses explicit activation. Detailed slices stay folded
until requested. Slices,
physical crosshairs and profiles preserve stored values. The volume display
uses documented interpolation, gradient lighting and an opacity transfer,
without evaluating transport or reconstruction. Keep the acquired diagnostic's
calibration/convergence limits and the CT-derived phantom's material assignment
visible. Source rights, arrays, coordinate conventions and hashes are curated
before assets enter the public directory; the release registry checks them.

The CT figure's white reference and recovered surfaces share three projective
radiograph lights on the book's paper background. Each image appears only on
its corresponding detector plane; there is no separate fitting-image strip.
Both static model posters also use paper backgrounds.
`reconstruction-radiographs/study.json` records the original
fixed acquisition poses as object-frame source positions, full detector cell
faces and homogeneous bottom-origin UV maps. The texture upload reverses PNG
rows exactly once. RGB identifies fitting views 1, 17 and 33; the coloured
display does not evaluate transport or estimate poses. Frustums retain their
physical scale. Anatomy/full-acquisition framing moves only the shared camera,
and narrow displays stack both panes. Coloured illumination applies to surface
mode; recorded volume rendering keeps its neutral scalar transfer function.

`RegistrationResults` and `AcquisitionResults` display the frozen application
study through native `RecordedPlot` charts, static radiograph panels and tables.
The discrete registration view selector hides no outcomes from the downloadable
record; all image arrangements appear without JavaScript and in print. Preserve
the transparent stress mask and distinguish outside-support model predictions
from observations. Keep geometric success separate from the unchanged failed
stationarity tests. Acquisition plots compare paired noise at a common photon
budget; selection does not consistently outperform the fixed policy.

`WorkedExample` displays the successful examples in Chapters 11–13 from their
recorded numerical data. `RecordedPlot` renders native SVG curves, candidate bars
and paired outcome markers; paired noise replicates are never joined as a time
series. `WorkedRaster` draws complete recorded detector arrays and selected
material slices into canvas panels with native HTML labels and controls. All
plot labels, controls, annotations and result tables use the book's sans-serif
UI font stack. Layout, spacing, tables and folded data links reuse the shared
application-result figure styles. No Matplotlib plate is embedded.

`tools/public/export-worked-example-arrays.py` verifies the accepted execution
records and preserves every array value at its original dtype. Material origins
identify the first sample centre: physical z = 0 interpolates planes 7 and 8
equally. Canvas row reversal places increasing physical y upwards. Colour windows
remain fixed across states, materials and replicates, while linked pointer and
keyboard readouts expose unclipped numerical values. Without JavaScript, the SVG
plots, acceptance table and numerical downloads remain available. The browser
performs display interpolation only; it does not run a transport or inverse solver.

The unchanged `pose-sensitivity` figure now belongs in Chapter 5; its introduction
anchor links to that location. The unchanged `figure-radiograph-noise` plate is in
Chapter 7, with its Appendix anchor retained. `SpectralRadiographs` replaces the
Chapter 8 spectrum-only illustration with three recorded CT-derived detector
expectations. Its explicit unequal channel windows do not support comparing
transmission by matching grey levels.
