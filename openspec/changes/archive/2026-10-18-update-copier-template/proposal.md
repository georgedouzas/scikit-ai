# Proposal

## Why

The project was generated from `copier-pdm-nox` 0.14.0. The template has since been renamed to
`copier-modern-python` and reached 2.0.2: properdocs instead of mkdocs, deptry/pydoclint/typos/Renovate,
a committed lock file, and an OpenSpec constitution. Updating now, before the library is redirected
(see change `redefine-library-scope`), means the new code is built on the current tooling and doesn't
need migrating later.

## What Changes

- Point `.copier-answers.yml` at `gh:georgedouzas/copier-modern-python.git` and update from `0.14.0` to `2.0.2`.
- Answer the questions added since 0.14.0: `git_provider=GitHub`, `package_manager=PDM`,
  `project_layout=library`, `include_dockerfile=false`, `spec_tool=openspec`, `publish_pypi=true`.
- Take the regenerated tooling: noxfile, `pyproject.toml` dev groups and tool config, CI workflows,
  pre-commit, Renovate, `properdocs.yml` (replacing `mkdocs.yml`), CONTRIBUTING, docs scaffolding.
- Merge the template's `openspec/config.yaml` `context:` and `openspec/constitution.md` with the
  OpenSpec setup already in the repo.
- Commit `pdm.lock` as the template now requires.
- Keep project content (README body, `src/skai`, tests, CHANGELOG): the update must not bring back
  template placeholders over it.

## Capabilities

### New Capabilities

None. This is a tooling change with no effect on library behavior (`skip_specs: true`).

### Modified Capabilities

None.

## Impact

- Developer workflow: `pdm docs|formatting|checks|tests|changelog|release` keep their names. The docs
  build moves to properdocs, and new checks (deptry, pydoclint, typos) may flag existing code.
- CI: GitHub workflows are regenerated, and Renovate opens dependency PRs.
- No public API change for `skai`.
