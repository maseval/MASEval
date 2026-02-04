# Seeding

The seeding module provides infrastructure for reproducible benchmark runs. Seeds are derived deterministically from hierarchical paths, ensuring that adding or removing components doesn't affect seeds for other components.

!!! tip "Guide available"

    For practical usage examples and best practices, see the [Seeding Guide](../guides/seeding.md).

The abstract base class that defines the seeding interface. Implement this to create custom seeding strategies.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/seeding.py){ .md-source-file }

::: maseval.core.seeding.SeedGenerator

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/seeding.py){ .md-source-file }

::: maseval.core.seeding.DefaultSeedGenerator

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/seeding.py){ .md-source-file }

::: maseval.core.seeding.SeedingError
