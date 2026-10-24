# Repository status and manuscript rights

Status checked through the authenticated GitHub repository API on 24 October
2026: this repository is **public**. Its current author manuscript and
supplementary PDFs can be retrieved without a private reviewer-access grant.
Earlier status notes recorded a private repository on 24 September; that is a
historical observation, not the current access state. This update does not
change repository visibility or permissions.
The current branch contains the 24 October 2026 Applied Energy author PDFs,
numerical code, author-generated result tables and figure data. It does not
contain the current manuscript TeX source package or standalone manuscript
figures. The current R3 release note records the exact deposited PDF hashes (23-page main text and 55-page SI).

Removing source assets from the current branch did **not** remove them from
older public commits or tags. This revision does not rewrite Git history.
Authors concerned about historical source exposure should review repository
history with the institution before choosing any separate history-management
action. The current compiled PDFs are publicly accessible and remain author
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
No repository DOI is assigned. The manuscript's data and code statements point to this public URL and the release note identifying the corresponding PDFs. The repository contains exact saved results plus
instructions distinguishing arithmetic verification from complete AC reruns.

## Author checks before journal submission

- Recognize that the repository and its existing Git history are public. No visibility change or history rewrite is performed by this update.
- Confirm all authors approve any planned public availability of the current draft PDFs
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
