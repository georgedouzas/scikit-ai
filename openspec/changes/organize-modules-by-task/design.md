# Design

## Context

Components live in `skai/{text,image,audio}/_*.py` and re-export through each package `__init__`. Their behavior is
specified per input type. See proposal.md for the motivation.

## Goals / Non-Goals

**Goals:**
- One module per task family, each component in exactly one place, with the constitution's import direction intact
  (`core` → `backends` → task-family packages).

**Non-Goals:**
- Renaming classes or changing behavior.
- Compatibility aliases for the old paths. 0.2 is unreleased, and an alias would put each name in two places.

## Decisions

- **Move files with `git mv`.** The implementation modules move unchanged into their new packages, for example
  `text/_classifier.py` → `classification/_text.py` and `image/_classifier.py` → `classification/_image.py`. Moving
  keeps history, and module names inside a package become the input type where a package holds several input types.
- **`skai.text` keeps its name** for text-to-text components. It matches `sklearn.feature_extraction.text`, and the
  alternatives (`rewriting`, `language`) describe extraction poorly.
- **Specs move with REMOVED plus ADDED deltas.** OpenSpec has no move operation, so each requirement is removed from
  its input-type capability and added verbatim to the task-family capability. The emptied spec files are deleted after
  archive.
- **Generators keep output-named classes.** `ImageGenerator` is named for what it makes, and its input is whatever its
  `X` is. The naming rule in `module-layout` states this exception.

## Risks / Trade-offs

- [Users of the unreleased 0.2 branch break] → The migration page lists the new import paths.
- [`skai.text` changes meaning between commits] → Its new contents are disjoint from the old ones, so a stale import fails with `ImportError` and doesn't silently do something else.
