# Repository status and manuscript rights

Status checked on 24 September 2026: this GitHub repository is **public**.
The current branch contains compiled author manuscript and Supporting
Information PDFs, numerical code, author-generated result tables and figure
data. It does not contain the current manuscript TeX source package or
standalone manuscript figures.

Removing source assets from the current branch did **not** remove them from
older public commits or tags. This revision does not rewrite Git history.
Authors concerned about historical source exposure should review repository
history with the institution before choosing any separate history-management
action. The current compiled PDFs are publicly readable, but remain author
manuscripts rather than accepted or publisher versions of record.

## Reuse terms

The scoped terms in [LICENSING.md](../LICENSING.md) and the
[per-file register](../provenance/LICENSE_MANIFEST.csv) apply:

- Original simulation and analysis code: MIT.
- Author-owned calculated data and explanatory documentation: CC BY 4.0.
- Manuscript and SI PDFs: all rights reserved under the manuscript notice.
- Third-party measurements, networks, metadata and dependencies: their own terms.

These are different rights for different materials; a public GitHub page
does not change the manuscript licence or grant rights in upstream data.
No repository DOI is assigned. The manuscript's data and code statements
point to this URL, and the repository contains exact saved results plus
instructions distinguishing arithmetic verification from complete AC reruns.

## Author checks before a Nature Communications submission

- Confirm all authors approve public availability of the current draft PDFs
  and understand that older Git history may contain withdrawn source files.
- Verify the submitted manuscript and SI PDFs match `paper/` and cite a fixed
  repository commit or release if the editorial system requests an immutable
  version.
- Decide the journal-specific AI-use declaration separately; no declaration
  was fabricated in this repository update.
- Check third-party dataset and figure permissions, author details, funding,
  conflict statements and the journal's current submission form.

The scientific archive does not include the third-party raw profiles and
network models or all local intermediate caches. Full model reruns therefore
require separately obtained source inputs and cache reconstruction. The
deposited record is suitable for inspecting code, source data and saved-result
arithmetic; it is not presented as a one-command independent field validation.
