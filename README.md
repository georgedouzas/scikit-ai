[black badge]: <https://img.shields.io/badge/%20style-black-000000.svg>
[black]: <https://github.com/psf/black>
[docformatter badge]: <https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg>
[docformatter]: <https://github.com/PyCQA/docformatter>
[ruff badge]: <https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json>
[ruff]: <https://github.com/charliermarsh/ruff>
[mypy badge]: <http://www.mypy-lang.org/static/mypy_badge.svg>
[mypy]: <http://mypy-lang.org>
[mkdocs badge]: <https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat>
[mkdocs]: <https://squidfunk.github.io/mkdocs-material>
[version badge]: <https://img.shields.io/pypi/v/scikit-ai.svg>
[pythonversion badge]: <https://img.shields.io/pypi/pyversions/scikit-ai.svg>
[downloads badge]: <https://img.shields.io/pypi/dd/scikit-ai>
[gitter]: <https://gitter.im/scikit-ai/community>
[gitter badge]: <https://badges.gitter.im/join%20chat.svg>
[discussions]: <https://github.com/georgedouzas/scikit-ai/discussions>
[discussions badge]: <https://img.shields.io/github/discussions/scikit-ai/scikit-ai>
[ci]: <https://github.com/georgedouzas/scikit-ai/actions?query=workflow>
[ci badge]: <https://github.com/scikit-ai/scikit-ai/actions/workflows/ci.yml/badge.svg?branch=main>
[doc]: <https://github.com/georgedouzas/scikit-ai/actions?query=workflow>
[doc badge]: <https://github.com/scikit-ai/scikit-ai/actions/workflows/doc.yml/badge.svg?branch=main>

# scikit-ai

[![ci][ci badge]][ci] [![doc][doc badge]][doc]

| Category          | Tools    |
| ------------------| -------- |
| **Development**   | [![black][black badge]][black] [![ruff][ruff badge]][ruff] [![mypy][mypy badge]][mypy] [![docformatter][docformatter badge]][docformatter] |
| **Package**       | ![version][version badge] ![pythonversion][pythonversion badge] ![downloads][downloads badge] |
| **Documentation** | [![mkdocs][mkdocs badge]][mkdocs]|
| **Communication** | [![gitter][gitter badge]][gitter] [![discussions][discussions badge]][discussions] |

## Introduction

A unified AI library that brings together classical Machine Learning, Reinforcement Learning, and Large Language Models under a
consistent and simple interface.

It mimics the simplicity of [scikit-learn](https://scikit-learn.org/stable)’s API and integrates with its ecosystem, while also
supporting libraries like [TorchRL](https://docs.pytorch.org/rl/stable/index.html), [oumi](https://oumi.ai/), and others.

## Installation

For user installation, `scikit-ai` is currently available on the PyPi's repository, and you can
install it via `pip`:

```bash
pip install scikit-ai
```

Development installation requires to clone the repository and then use [PDM](https://github.com/pdm-project/pdm) to install the
project as well as the main and development dependencies:

```bash
git clone https://github.com/georgedouzas/scikit-ai.git
cd scikit-ai
pdm install
```

## Usage

We aim to provide a simple and user-friendly API for working with AI models and functionalities.

### Classification

Let’s start with a basic text classification example:

```python
X = ['This is a positive review.', 'This is a negative review.']
y = [1, 0]
```

Now, create a k-shot classifier:

```python
from skai.llm import OpenAIClassifier

clf = OpenAIClassifier()
clf.fit(X, y)
clf.predict([
    'I absolutely loved this movie!',
    'The product was terrible and broke immediately.'
])
```

By default, the classifier uses reasonable k-shot settings. You can inspect them:

Number of examples used in the prompt:

```python
print(clf.k_shot_)
```

Instructions given to the model:

```python
print(clf.instructions_)
```

The complete prompt sent to the language model:

```python
print(clf.prompt_)
```

You can also customize the classifier in detail by adjusting its parameters. For more options and examples, please consult the
full API documentation.
