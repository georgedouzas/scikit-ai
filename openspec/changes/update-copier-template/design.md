# Design

## Context

`.copier-answers.yml` records `_commit: 0.14.0` and `_src_path: gh:georgedouzas/copier-pdm-nox.git`.
The template repository has been renamed (GitHub redirects the old URL) and now asks extra questions.
`openspec init --tools claude` has already been run in this repo, creating `openspec/config.yaml` and
`.claude/` skills/commands. The template also ships `openspec/config.yaml` and `openspec/constitution.md`.

## Goals / Non-Goals

**Goals:**
- Bring the repo to template 2.0.2 with a clean `copier update` path for future releases.
- Have all template quality gates (`pdm checks`, `pdm tests`, `pdm docs build`) pass on the existing code.

**Non-Goals:**
- Changing library code beyond what the new gates require. The library rewrite is `redefine-library-scope`.
- Switching from PDM to uv.

## Decisions

- **Use `copier update`, not regenerate-and-diff.** A 3-way merge keeps local edits and keeps the answers
  file valid for later updates. Run `copier update --trust --vcs-ref 2.0.2 --data ...`, passing the new
  answers explicitly so nothing falls back to a default without notice. The alternative was copying a fresh
  render and diffing by hand. That loses the merge base, so the next update would be just as painful.
- **Rewrite `_src_path` first.** Editing the answers file to the new repo name before updating avoids relying
  on GitHub's rename redirect.
- **Keep PDM.** This keeps the current workflow and the `pdm <script>` entry points the docs refer to. uv is
  an available answer later if wanted.
- **OpenSpec config merge.** Keep the template's `context:` block (it binds `constitution.md`) in
  `openspec/config.yaml`, and keep the `.claude/` files that `openspec init` generated. Fill in the
  constitution's Project Profile once the new scope in `redefine-library-scope` is accepted.
- **Fix new-gate failures minimally.** deptry, pydoclint and typos findings on the current code get small
  fixes or targeted ignores. The current modules (`skai.llm`, `skai.base.rl`) are replaced by the next change,
  so refactoring them is wasted effort.

## Risks / Trade-offs

- [Large merge, many `.rej` files across 0.14 → 2.0] → Run on a dedicated branch, resolve conflicts file by file, and review `git diff` before committing.
- [The 0.14.0 tag may be missing or renamed upstream, which breaks the 3-way base] → Confirm the tag exists with `git ls-remote --tags`. If it doesn't, fall back to `copier recopy` and re-apply local changes by hand.
- [mkdocs → properdocs breaks docs URLs or plugins] → Build the docs locally and check the GitHub Pages deploy on the first CI run.

## Migration Plan

Single PR. Rollback is reverting the merge commit. `.copier-answers.yml` then points back to 0.14.0.
