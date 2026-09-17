# Manuscript sharing and reuse permissions

Status checked on 17 September 2026. This is a private author-review repository. This document neither makes the repository public nor grants a reuse licence.

## Manuscript source and research reproducibility are different deposits

The `paper/` directory contains the current author manuscript, SI, and compilation dependencies. It is retained here for private collaboration. Access to this repository is not a public preprint release, and the deposited PDF is not a publisher's version of record.

A future public reproducibility deposit can provide numerical code, author-generated result tables, figure data, and instructions without distributing the manuscript's TeX files. RSC's data-sharing guidance concerns the materials needed to support and reproduce the results; it does not list publication of the manuscript's TeX source as a requirement.

RSC permits preprint sharing subject to its prior-publication policy. Sharing rights for an accepted manuscript or publisher's version depend on the publication route and licence. Before releasing a paper version, confirm the applicable version, all authors' approval, third-party permissions, institutional requirements, and any patent considerations.

**Do not simply switch this repository to public if manuscript-source publication is unwanted.** Git history already contains manuscript sources. Deleting `paper/` in a later commit would not remove those historical copies. Prepare a separately reviewed, clean public deposit, or agree on an appropriate history-management plan first. No history rewrite or visibility change is part of this update.

## Proposed licensing split — pending author approval

There is no single licence that every research paper or research-code repository uses. The following split is recommended for this deposit, but has not been applied:

| Material | Proposed treatment | Practical effect |
|---|---|---|
| Original numerical and reproduction code | MIT | Permits modification, redistribution, and commercial use with the required copyright and licence notices |
| Author-owned calculated result tables and explanatory documentation outside `paper/` | CC BY 4.0 | Permits sharing and adaptation, including commercial reuse, with attribution, a licence link, and notice of changes |
| Code snippets in explanatory documentation | MIT | Keeps executable examples under the same terms as code |
| Main manuscript, SI, their TeX source, and manuscript figures | Not included in the proposed grant | Decide separately in accordance with the authors' publication plan and applicable publishing agreement |
| Third-party measurements, networks, templates, bibliography styles, and other third-party material | Original terms only | This project cannot grant rights it does not hold |

The MIT and CC BY proposals do not restrict reuse to academic or non-commercial purposes. CC BY 4.0 cannot be revoked for recipients who comply with its terms. Anyone preferring a non-commercial restriction should resolve that choice before release, rather than treating it as an interchangeable wording change.

For an open-access journal article, RSC currently offers CC BY and CC BY-NC options, subject to the applicable institutional or funder requirements. That article-licence choice is separate from the code licence. No publisher licence has been selected by this repository update.

## Before public release or journal submission

- [ ] Authors approve whether the manuscript PDF and/or source will be shared publicly.
- [ ] Rights holders approve the code/data licensing split, including permission for commercial reuse.
- [ ] Confirm the ownership and applicable terms of each material; retain all third-party notices and exclusions.
- [ ] Provide the editors and reviewers with working access to the materials described in the manuscript's data-availability statement. A private URL alone does not provide that access.
- [ ] Prepare and check the intended public file set and history before changing visibility or publishing another repository.
- [ ] Add the approved licence texts and exact file/directory scope; update the README and citation metadata consistently.
- [ ] Cite a fixed release or commit. A repository DOI is not currently assigned; RSC accepts a code URL when a DOI is unavailable and recommends durable archiving.
- [ ] Recheck article-version sharing rights when the manuscript is accepted or published.

## Official guidance consulted

- [RSC: prior publication and sharing of preprints](https://www.rsc.org/publishing/journals/processes-and-policies)
- [RSC: licences, copyright and permissions](https://www.rsc.org/publishing/journals/processes-and-policies/licences-copyright-and-permissions)
- [RSC: data sharing and software citation](https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/data-sharing)
- [GitHub: licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
- [Choose a License: mixed projects and non-software material](https://choosealicense.com/non-software/)
- [MIT licence text and conditions](https://choosealicense.com/licenses/mit/)
- [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/)

This checklist records the release decisions still needed; it is not a substitute for the rights holders' approval or institution-specific advice.
