"""Run the examples of the docstrings and of the README."""

import doctest
import re
from pathlib import Path

import pytest

import skai
import skai.backends
import skai.classification
import skai.core
import skai.embedding
import skai.extraction
import skai.generation
import skai.pipeline
import skai.presets
import skai.recipes
import skai.segmentation

COMPONENT_MODULES = [
    skai.classification,
    skai.extraction,
    skai.embedding,
    skai.segmentation,
    skai.generation,
    skai.pipeline,
]
MODULES = [skai, skai.backends, skai.core, skai.presets, skai.recipes, *COMPONENT_MODULES]
README = Path(__file__).parents[1] / 'README.md'


def find_examples() -> list[doctest.DocTest]:
    """Find the docstring examples of every public module and every name it exports."""
    finder = doctest.DocTestFinder(exclude_empty=True)
    tests = [test for module in MODULES for test in finder.find(module, module.__name__)]
    for module in MODULES:
        for name in getattr(module, '__all__', []):
            tests += finder.find(getattr(module, name), f'{module.__name__}.{name}', globs={})
    unique = {test.name: test for test in tests if test.examples}
    return sorted(unique.values(), key=lambda test: test.name)


@pytest.mark.parametrize('test', find_examples(), ids=lambda test: test.name)
def test_docstring_examples_run(test):
    """Run every docstring example and compare its output."""
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    runner.run(test)
    assert runner.failures == 0, f'{runner.failures} example(s) failed in {test.name}'


def test_every_public_component_has_an_example():
    """Give every public component, primitive and the dummy backend a runnable example."""
    names = {test.name for test in find_examples()}
    components = [f'{module.__name__}.{name}' for module in COMPONENT_MODULES for name in module.__all__]
    assert {*components, 'skai.backends.DummyBackend'} <= names


def test_presets_and_recipes_have_module_examples():
    """Give the presets and recipes modules a runnable example."""
    names = {test.name for test in find_examples()}
    assert {'skai.presets', 'skai.recipes'} <= names


def test_readme_examples_run():
    """Run every Python block of the README in order."""
    blocks = re.findall(r'```python\n(.*?)```', README.read_text(), flags=re.DOTALL)
    assert blocks
    namespace: dict = {}
    for block in blocks:
        exec(block, namespace)  # noqa: S102 # nosec B102 - runs the README examples, which the repository controls
