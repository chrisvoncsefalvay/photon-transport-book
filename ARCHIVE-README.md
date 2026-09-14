# Differentiable Photon Transport — Volume I: Foundations

Chris von Csefalvay · Version 1.0.0 · 14 September 2026

Edition DOI: [10.5281/zenodo.22758650](https://doi.org/10.5281/zenodo.22758650).
Read online at [photontransport.com](https://photontransport.com/).
Repository: [photon-transport-book](https://github.com/chrisvoncsefalvay/photon-transport-book).

This instructional book develops differentiable photon transport through X-ray
and C-arm imaging, with executable NVIDIA Warp/CUDA examples. The two archives
preserve the same edition in complementary forms:

- `photon-transport-book-1.0.0-source.zip` contains the public manuscript, website
  source, computational recipes, figure assets, dependency lockfiles,
  citation metadata and licence notices.
- `photon-transport-book-1.0.0-site.zip` contains the built website in `site/`,
  including its interactive figures, numerical display data and source listings.
  Licence notices, this README and the edition's citation metadata accompany it.

To read the website archive, extract it and run this command from its top-level
directory:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory site
```

Open <http://127.0.0.1:8000/> in a browser. Serving the files over HTTP allows
JavaScript modules and recorded data to load. External references still require
internet access. The static website needs no GPU; rerunning the computational
experiments requires the environment described in the source archive's README.

`CITATION.cff` and `release-metadata.json` identify the edition. The source
manifest and website snapshot identify the authoring commit and the exact source
listings used by the site. The authoring commit and a public repository commit
are distinct identifiers. Numerical figure data retain their accompanying
provenance records and scientific qualifications.

Book text and editorial content are licensed under CC BY-NC 4.0
(`LICENSE-CONTENT`). Code is licensed under Apache License 2.0 (`LICENSE-CODE`).
Third-party assets and derived data retain their own licences; consult
`ATTRIBUTIONS.md` and their provenance records. The book's licence does not
replace those terms.

`SHA256SUMS` accompanying the deposit records the checksums of both ZIP files and
this README. The deposit preserves a web edition; it does not include a PDF, raw
medical volumes, private research records or model weights.
