# Spec Delta

## Purpose

Keeps code written for 0.2 working for one minor release while it moves to the general components.

## ADDED Requirements

### Requirement: Deprecated 0.2 names
Every public component name of 0.2 SHALL stay importable from its 0.2 module for the 0.3 releases, as a thin subclass of the general component configured as before. Constructing it SHALL emit one `FutureWarning` that names the replacement and the release that removes it.

#### Scenario: Old classifier
- **WHEN** `from skai.classification import ImageClassifier` is used and `ImageClassifier(labels=['cat', 'dog'])` is constructed
- **THEN** a `FutureWarning` names `Classifier(input='image')`, and the object behaves like it

#### Scenario: Old modules
- **WHEN** `from skai.text import TextSummarizer` or `from skai.description import ImageDescriptor` is used
- **THEN** the import succeeds with a `FutureWarning` naming the replacement

### Requirement: Removal in 0.4
The deprecated names and the `skai.text` and `skai.description` modules SHALL be removed in 0.4.

#### Scenario: Documented removal
- **WHEN** a user reads the migration page
- **THEN** it lists every 0.2 name, its replacement, and that it is removed in 0.4
