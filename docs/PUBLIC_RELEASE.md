# Manuscript sharing and reuse permissions

Status checked on 17 September 2026. This remains a private author-review repository. The authors authorized the scoped licences now applied in [LICENSING.md](../LICENSING.md). No visibility change, history rewrite or public article release is part of this update.

## Manuscript source and research reproducibility are different deposits

The `paper/` directory now contains only the compiled author manuscript, SI and a sharing notice. All 27 manuscript compilation/source assets have been withdrawn from the current tree. The numerical code, results and figure-data tables remain. The local author master and submission source packages are unchanged. Access to this private repository is not a public preprint release, and the deposited PDFs are not publisher versions of record.

A future public reproducibility deposit can provide numerical code, author-generated result tables, figure data, and instructions without distributing the manuscript's TeX files. RSC's data-sharing guidance concerns the materials needed to support and reproduce the results; it does not list publication of the manuscript's TeX source as a requirement.

RSC permits preprint sharing subject to its prior-publication policy. Sharing rights for an accepted manuscript or publisher's version depend on the publication route and licence. Before releasing a paper version, confirm the applicable version, all authors' approval, third-party permissions, institutional requirements, and any patent considerations.

**Do not simply switch this repository to public if manuscript-source publication is unwanted.** Older commits and tags still contain manuscript sources despite their removal from the current tree. Keep the existing URL; agree on an appropriate history-management plan before any visibility change. No history rewrite, deletion of historical tags or visibility change has been performed.

## Applied licensing split

The author authorized the following split. Exact paths are recorded in [the per-file register](../provenance/LICENSE_MANIFEST.csv):

| Material | Applied treatment | Practical effect |
|---|---|---|
| Original numerical and reproduction code | MIT | Permits modification, redistribution, and commercial use with the required copyright and licence notices |
| Author-owned calculated result tables and explanatory documentation outside `paper/` | CC BY 4.0 | Permits sharing and adaptation, including commercial reuse, with attribution, a licence link, and notice of changes |
| Code snippets in explanatory documentation | MIT | Keeps executable examples under the same terms as code |
| Main manuscript and SI PDFs | All rights reserved; excluded from open code/data grants | Retained for authorized review; article publication rights are decided separately |
| Manuscript TeX, bibliography/style files, standalone figures and publisher compilation assets | Removed from the current tree | Historical copies remain outside the new open grants |
| Third-party measurements, networks, templates, bibliography styles, and other third-party material | Original terms only | This project cannot grant rights it does not hold |

The MIT and CC BY grants do not restrict reuse of the covered author-owned materials to academic or non-commercial purposes. CC BY 4.0 cannot be revoked for recipients who comply with its terms. Third-party rights are excluded: for example, the CSIRO/GridQube collection has its own CC BY-NC-SA 4.0 conditions.

For an open-access journal article, RSC currently offers CC BY and CC BY-NC options, subject to the applicable institutional or funder requirements. That article-licence choice is separate from the code licence. No publisher licence has been selected by this repository update.

## Before public release or journal submission

- [x] Remove manuscript compilation sources from the current tree while retaining the repository URL and local author master.
- [x] Apply the author-authorized code/data split and retain third-party exclusions and original notices.
- [x] Record terms for every current file; keep manuscript PDFs outside the open grants.
- [ ] Decide whether the manuscript PDFs will be shared publicly; no public article release is authorized by this update.
- [ ] Provide the editors and reviewers with working access to the materials described in the manuscript's data-availability statement. A private URL alone does not provide that access.
- [ ] Prepare and check the intended public file set and history before changing visibility or publishing another repository.
- [x] Add the approved licence texts and exact file/directory scope; update the README and citation metadata consistently.
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

This checklist distinguishes completed repository changes from access and publication decisions still needed.
