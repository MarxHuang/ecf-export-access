# Deposited revision: nexus-submission-2026-10-13

This revision synchronizes the compiled author manuscript and Supplemental information with the Nexus submission master dated 13 October 2026. The PDFs contain 25 and 54 pages, respectively. The update changes presentation, terminology and explanations; it does not introduce new simulations or numerical results.

## Materials synchronized

- `paper/main.pdf` matches the current submission manuscript byte-for-byte.
- `paper/supporting_information.pdf` matches the current Supplemental information byte-for-byte. The filename remains unchanged so existing links still work.
- All eight submission source-data workbooks already match deposited files byte-for-byte. They are not duplicated or replaced. The exact correspondence and SHA256 hashes are recorded in `provenance/RELEASE_CHECK_20261013_NEXUS.json`.
- Figure references in the data guide use the current main and supplemental numbering. Historical directory names remain unchanged for reproducibility.
- `CITATION.cff` identifies this deposited revision. It does not assert publication or assign a DOI.

## Verification

The release check reports file-integrity verification, the saved Australian energy balances, the original three-schedule and expanded 12-schedule comparisons, and the primary-case dry run. These checks use saved results; they do not rerun a power-flow model or constitute a new experimental result.

Original numerical implementations, experiment protocols, saved results and figure-data bytes are unchanged. Previous release notes and checks remain available as historical records. Use the [figure-data guide](FIGURE_DATA.md) to select the current plotting inputs and the [reproduction instructions](REPRODUCING.md) for model execution requirements.

## Sharing and rights

Repository visibility and licences are unchanged. Original code remains under MIT, author-owned calculated data and documentation under CC BY 4.0, and compiled manuscript PDFs under the existing all-rights-reserved notice. Third-party terms continue to apply to source data and networks.

This update does not add manuscript TeX, Overleaf packages, bibliography/style files, standalone manuscript figures, logos, cover letters, interest declarations or other submission-only documents. Older commits may contain previously withdrawn sources; this update does not rewrite history. See [repository access and rights](PUBLIC_RELEASE.md) before arranging reviewer access or any public release.
