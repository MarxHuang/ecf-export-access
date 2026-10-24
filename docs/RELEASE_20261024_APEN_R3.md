# Author revision apen-preflight-2026-10-24-r3

This deposit corresponds to APEN_PREFLIGHT_20261024_R3. It replaces the stable
main/SI PDFs with this local author revision. R2 and earlier release notes
remain historical records. This is not journal submission, acceptance or a
publisher version of record.

## Current files

- `paper/main.pdf`: 23 pages, SHA256 `bf863a8709a62afb6fba97b52d8e661c2593cba118e2d9a41f3b4867aa10e629`.
- `paper/supporting_information.pdf`: 55 pages, SHA256 `69b435f580f7dc950ba857c3e9535c890e7cc04ee6dfd2799cae608fbfafc81a`.

## Change and evidence scope

Figures and tables now use controlled here placement at complete-paragraph
boundaries, preventing a deferred float from separating the parts of a sentence.
The resulting page counts are 23 and 55; text and artwork were not reduced to
retain the old page counts. Figure 4's caption explicitly identifies both
panels (c) and (d) as gain-0.5 results, consistent with the retained plotting input.
The figure-data map distinguishes Figure 8/S5 plotting inputs from the
bounded-record, PV-capacity and nested-section tables and documents historical
INDEX/SOURCE_INFO labels. No workbook was recalculated or altered.

Models, formulas, proofs, numerical results, positive/negative outcomes and
all 14 artwork PDFs are unchanged. No simulation or AC solve was run for this
revision. An independently extracted local source package recompiles all
23 main-text and 55 SI pages with identical text and rendered page pixels.
All eight workbooks retain their prior bytes and hashes.

`provenance/RELEASE_CHECK_20261024_APEN_R3.json` records PDF identities and
workbook correspondence. `FIGURE_DATA.md` records current plotting and table sources.

## Public scope and rights

The repository is public. This update changes no visibility, permissions,
licence or historical commit. Existing scoped rights remain in effect:
original numerical code is MIT, author-owned calculated data/documentation
are CC BY 4.0, and manuscript PDFs remain all rights reserved. Third-party
raw inputs remain with their original distributors.

No cover letter, editorial correspondence, author declaration draft,
manuscript TeX/Overleaf source ZIP, institutional logo or publisher compilation
asset is added. Earlier commits can retain historical sources; no history
rewrite was performed.
