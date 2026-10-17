# Tasks

## 1. Move the modules

- [x] 1.1 Move the component modules into `skai/{classification,description,transcription,text,embedding,generation}` with `git mv`, write each package `__init__` with an explicit `__all__`, delete `skai/image` and `skai/audio`, and verify `ruff`, `mypy` and `interrogate` pass on `src`
- [x] 1.2 Update the docstring examples, the `skai` package docstring and the entry points, and verify the docstring example test passes

## 2. Tests and docs

- [x] 2.1 Move the component tests to `tests/{classification,description,transcription,text,embedding,generation}`, update every import, add tests for the `module-layout` scenarios, and verify `pdm tests` passes on all supported Python versions
- [x] 2.2 Update the README, the user guide pages, the migration page and the constitution's Project Profile to the new modules, and verify `pdm docs build` passes in strict mode and the README test passes

## 3. Gate

- [x] 3.1 Run `pdm formatting`, `pdm checks`, `pdm docs build` and `pdm tests` in order, and verify they all pass

## Workflow follow-up

- Archive the change, which retires the `text-components`, `image-components` and `audio-components` specs.
