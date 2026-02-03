# Seeding

The seeding module provides infrastructure for reproducible benchmark runs. Seeds are derived deterministically from hierarchical paths, ensuring that adding or removing components doesn't affect seeds for other components.

!!! tip "Guide available"

    For practical usage examples and best practices, see the [Seeding Guide](../guides/seeding.md).

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/seeding.py){ .md-source-file }

## SeedGenerator

The abstract base class that defines the seeding interface. Implement this to create custom seeding strategies.

::: maseval.core.seeding.SeedGenerator

## DefaultSeedGenerator

The default implementation using SHA-256 hashing. Provides additional convenience methods like `child()` for hierarchical namespacing.

::: maseval.core.seeding.DefaultSeedGenerator

## SeedingError

Exception raised when seeding is misconfigured or a provider doesn't support seeding.

::: maseval.core.seeding.SeedingError
