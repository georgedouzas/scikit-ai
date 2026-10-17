# Tasks

## 1. Template update

- [x] 1.1 Set `_src_path` in `.copier-answers.yml` to `gh:georgedouzas/copier-modern-python.git` and verify `git ls-remote --tags` on it lists `0.14.0` and `2.0.2`
- [x] 1.2 Run `copier update --trust --vcs-ref 2.0.2` with `git_provider=GitHub package_manager=PDM project_layout=library include_dockerfile=false spec_tool=openspec publish_pypi=true`, then verify `.copier-answers.yml` shows `_commit: 2.0.2` and the new answers
- [x] 1.3 Resolve all merge conflicts and `.rej` files, keeping project content (README body, `src/`, `tests/`, CHANGELOG), and verify `find . -name '*.rej'` returns nothing
- [x] 1.4 Merge `openspec/config.yaml` (template `context:` plus init-generated settings) and keep `openspec/constitution.md`, then verify `openspec validate --all` passes

## 2. Tooling gates

- [x] 2.1 Run `pdm lock && pdm install` and commit `pdm.lock`, verifying a clean install from the lock
- [x] 2.2 Run `pdm formatting` and `pdm checks` (ruff, mypy, deptry, pydoclint, typos), fix findings or add targeted ignores, and verify all checks pass
- [ ] 2.3 Run `pdm tests` and verify the existing suite passes on all supported Python versions
- [x] 2.4 Run `pdm docs build` with properdocs, verify the site builds, and confirm `mkdocs.yml` is gone

## 3. Integration

- [ ] 3.1 Push the branch and verify the regenerated GitHub CI and docs workflows pass
