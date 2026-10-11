# Releasing the calculator

The version lives in three places that must agree: `calculator/version.js`, `py/pyproject.toml`
(and `py/evidential_ceiling/__init__.py`), and `CITATION.cff`. The website shows
`calculator/version.js`, and its version must match this repository's release tag.

## Each release

1. Bump the version in the files above. Run both test suites:
   `node --test 'calculator/*.test.js'` and `PYTHONPATH=py python -m unittest discover -s py/tests`.
2. Commit, tag `vX.Y.Z`, push the tag, and publish a GitHub release from it.
3. In the website repository: `npm run sync:ceiling`, `npm run og:ceiling`, build, commit.

## Zenodo DOI (one-time setup, then automatic)

1. Sign in at https://zenodo.org with GitHub (Account → Linked accounts → GitHub).
2. Go to https://zenodo.org/account/settings/github/, press "Sync now", and switch
   `hackwither/evidential-limits` on.
3. Publish a GitHub release (step 2 above). Zenodo archives it and mints two DOIs: one for this
   version and a "concept" DOI that always resolves to the latest version. `.zenodo.json` supplies
   the title, creator, ORCID, licence and the link to the paper's DOI.
4. Put the concept DOI in:
   - `CITATION.cff` (an `identifiers:` entry with `type: doi`),
   - the website's `@software` BibTeX on `/evidential-ceiling/` (replace `TODO-zenodo`),
   - optionally a DOI badge in `README.md`.
5. Check the Zenodo record's metadata once; later releases inherit it.

## PyPI (when wanted)

```bash
cd py
python -m pip install build twine
python -m build
python -m twine upload dist/*
```

Use a PyPI API token scoped to the `evidential-ceiling` project, or set up trusted publishing
from a GitHub Actions workflow instead of a token.
