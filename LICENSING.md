# Licensing scope

Applied with author authorization on 17 September 2026. This is a mixed-material
repository, not a dual-licensed work in which every file can be used under either
MIT or CC BY. Repository access and reuse permission are separate: applying these
terms does not make this private repository public.

## Scope by material

| Material | Terms |
|---|---|
| Original Python files, including tests, numerical implementations and analysis/plotting tools, wherever stored | MIT |
| `reproduction/pyproject.toml`, `.gitignore`, `.gitattributes` | MIT |
| Author-generated CSV/XLSX tables, numerical JSON records, losslessly compressed results, YAML experiment protocols, date lists and source-hash manifests | CC BY 4.0, limited to the contributors' own rights |
| Original Markdown documentation and `CITATION.cff` | CC BY 4.0; executable code examples are additionally available under MIT |
| `paper/main.pdf` and `paper/supporting_information.pdf` | All rights reserved; see the manuscript notice |
| The two copied CSIRO collection/Schema.org metadata records named below | Original source terms only, not a licence grant by this project |
| Standard licence texts under `LICENSES/` | Retained as legal notices; not relicensed as project-authored research |

The complete current path list is in
[`provenance/LICENSE_MANIFEST.csv`](provenance/LICENSE_MANIFEST.csv).
This register takes precedence over a broad directory description if a directory
contains more than one material type. A copied third-party notice always retains
its own effect. Gzip changes storage only, not the rights in the expanded content.

Copyright in the original code and author-owned research materials is held by the
ECF export-access contributors, 2026, to the extent of their respective rights.
The grants do not cover rights held by institutions or other parties that the
contributors are not entitled to license.

## Reusing code and results

MIT permits software use, modification and redistribution, including commercial
use, subject to retaining its copyright and licence notices.

CC BY 4.0 permits sharing and adaptation, including commercial reuse, with
attribution, a licence link and an indication of changes. Suggested attribution:
“ECF export-access contributors, *Increasing electricity exports through
delivery-informed allocation in distribution networks: research materials*,
repository revision [commit or tag], https://github.com/MarxHuang/ecf-export-access,
CC BY 4.0.” Use the author list in `CITATION.cff` for scholarly citation.
Neither a publication DOI nor a repository DOI is assigned here.

These terms cover the authors' original contributions to calculated outputs and
data organization. They do not license the underlying Ausgrid measurements,
GridQube networks, copied metadata, or any upstream protected material that may
be needed to reproduce the results. If reuse also involves upstream material or
an adaptation of it, its applicable terms must be met separately. Do not infer
permission for commercial use of an upstream dataset from the licence on our code
or original calculated outputs.

## Third-party exceptions

The following unmodified records are retained with their source notices:

- `reproduction/outputs/ees_transition/archive/091_data/external/csiro_australian_mv_lv_feeder_v1/CSIRO_COLLECTION_METADATA.json`
- `reproduction/outputs/ees_transition/archive/091_data/external/csiro_australian_mv_lv_feeder_v1/SCHEMA_ORG_METADATA.json`

They describe the collection at https://doi.org/10.25919/ghnz-bk28, whose stated
dataset licence is CC BY-NC-SA 4.0. No new MIT or CC BY grant is made for these
copied records. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Raw measurements, network models and third-party software distributions are not
bundled here. Install or obtain them under their respective original terms.

## Paper and publication rights

The two compiled paper PDFs are excluded from the code/data grants. The separate
notice in [LICENSES/LicenseRef-Manuscript.txt](LICENSES/LicenseRef-Manuscript.txt)
reserves their rights pending the authors' publication arrangements. Publisher
artwork, embedded fonts, third-party figures, logos and trademarks are not
relicensed by this project. `paper/README.md` is project documentation, not part
of the article, and is CC BY 4.0.

Manuscript compilation sources are absent from the current tree. Older commits
and tags still contain them. The scoped grants introduced here do not grant an
open licence to those historical manuscript files. A future public release must
resolve their visibility separately; see [PUBLIC_RELEASE.md](docs/PUBLIC_RELEASE.md).
