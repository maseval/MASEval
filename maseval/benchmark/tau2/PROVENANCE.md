# Tau 2 Benchmark - Provenance Documentation

This document tracks the source mapping between MASEval's Tau 2 implementation and the original tau2-bench codebase.

## Base Version

**tau2-bench v0.2.0** (commit `f8de30c`, 2025-10-06)

Repository: https://github.com/sierra-research/tau2-bench
License: MIT License
Copyright: (c) 2025 Sierra Research

## Why v0.2.0?

| Factor           | v0.2.0                              | HEAD (main)                    |
| ---------------- | ----------------------------------- | ------------------------------ |
| **Domain tools** | Complete                            | Identical (no changes)         |
| **Core tasks**   | 50/114/114 (airline/retail/telecom) | Same "base" split              |
| **Dependencies** | Simpler (no gymnasium)              | +gymnasium for RL training     |
| **Stability**    | Formal release                      | Includes experimental features |

## Component Mapping

### Core Infrastructure

| MASEval Component | tau2-bench Source      | Notes                           |
| ----------------- | ---------------------- | ------------------------------- |
| `data_loader.py`  | `data/tau2/domains/*/` | Downloads from v0.2.0 tag       |
| `utils.py`        | `tau2/utils/`          | Hashing, file loading utilities |

### Retail Domain

| MASEval Component          | tau2-bench Source                                | Notes                        |
| -------------------------- | ------------------------------------------------ | ---------------------------- |
| `domains/retail/models.py` | `src/tau2/domains/retail/data_model.py`          | Pydantic models for entities |
| `domains/retail/db.py`     | `src/tau2/domains/retail/data_model.py:RetailDB` | Database class               |
| `domains/retail/tools.py`  | `src/tau2/domains/retail/tools.py`               | All retail tools ported      |

### Airline Domain

| MASEval Component           | tau2-bench Source                                  | Notes                        |
| --------------------------- | -------------------------------------------------- | ---------------------------- |
| `domains/airline/models.py` | `src/tau2/domains/airline/data_model.py`           | Pydantic models for entities |
| `domains/airline/db.py`     | `src/tau2/domains/airline/data_model.py:AirlineDB` | Database class               |
| `domains/airline/tools.py`  | `src/tau2/domains/airline/tools.py`                | All airline tools ported     |

### Telecom Domain

| MASEval Component           | tau2-bench Source                                  | Notes                        |
| --------------------------- | -------------------------------------------------- | ---------------------------- |
| `domains/telecom/models.py` | `src/tau2/domains/telecom/data_model.py`           | Pydantic models for entities |
| `domains/telecom/db.py`     | `src/tau2/domains/telecom/data_model.py:TelecomDB` | Database class               |
| `domains/telecom/tools.py`  | `src/tau2/domains/telecom/tools.py`                | All telecom tools ported     |

### Benchmark Components

| MASEval Component                       | tau2-bench Source                             | Notes                         |
| --------------------------------------- | --------------------------------------------- | ----------------------------- |
| `Tau2Environment`                       | `src/tau2/environment/environment.py`         | Uses MASEval Environment base |
| `Tau2User`                              | `src/tau2/user/user_simulator.py`             | Uses MASEval User base        |
| `Tau2Evaluator._evaluate_environment`   | `src/tau2/evaluator/evaluator_env.py`         | DB state comparison           |
| `Tau2Evaluator._evaluate_actions`       | `src/tau2/evaluator/evaluator_action.py`      | Tool call verification        |
| `Tau2Evaluator._evaluate_communication` | `src/tau2/evaluator/evaluator_communicate.py` | Communication checks          |

### Improvements Adopted from HEAD

| Feature                     | Description                              | Rationale                                |
| --------------------------- | ---------------------------------------- | ---------------------------------------- |
| Evaluator termination logic | Explicit `AGENT_STOP`, `USER_STOP` check | More defensive than v0.2.0's reject-list |

## Data Files

Domain data is downloaded from the v0.2.0 tag:

```
https://github.com/sierra-research/tau2-bench/tree/v0.2.0/data/tau2/domains/
```

| Domain  | Files                               | Task Count (base split) |
| ------- | ----------------------------------- | ----------------------- |
| airline | db.json, tasks.json, policy.md      | 50                      |
| retail  | db.json, tasks.json, policy.md      | 114                     |
| telecom | db.toml, tasks.json, main_policy.md | 114                     |

## MASEval-Specific Additions

These components are new implementations that don't have direct tau2-bench equivalents:

| Component                     | Description                                        |
| ----------------------------- | -------------------------------------------------- |
| `Tau2Benchmark`               | Abstract benchmark base following MASEval patterns |
| `Tau2ToolBase`                | Base class for tools with MASEval tracing          |
| `compute_pass_at_k()`         | Pass@k metric computation                          |
| TraceableMixin integration    | Execution tracing for all components               |
| ConfigurableMixin integration | Configuration gathering                            |

## Intentional Implementation Differences from Original tau2-bench

MASEval's tau2 implementation does not exactly replicate the original tau2 benchmark implementation.

The following intentional differences exist between MASEval and the original tau2-bench:

### User Simulator Prompt

**Location:** `prompt_templates/user_simulator.txt`

| Difference                                                                | Rationale                                                   |
| ------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Explicit stop guidance ("When the agent confirms...END the conversation") | Prevents unnecessary conversation continuation              |
| "Do NOT ask unnecessary follow-up questions"                              | Reduces conversation length without affecting task validity |
| JSON output format required                                               | Structured parsing for MASEval framework                    |

These changes result in shorter conversations.

### Order ID Normalization (Retail Domain)

**Location:** `domains/retail/tools.py:66-68`

```python
if not order_id.startswith("#"):
    order_id = f"#{order_id}"
```

MASEval normalizes order IDs by adding the `#` prefix if missing. The original tau2-bench does not normalize, causing "Order not found" errors when LLMs omit the prefix. This is a minor leniency that improves usability without fundamentally changing task difficulty.

## Validation Strategy

1. **Deterministic evaluators** (env, action): Exact DB state hash match with upstream v0.2.0
2. **LLM-based evaluators** (communicate, NL): Within ±3% of upstream v0.2.0
3. **Contract tests**: Tool sequences produce identical state changes

## Key Characteristics

| Aspect          | Description                                       |
| --------------- | ------------------------------------------------- |
| Tools           | Real Python implementations with business logic   |
| State           | Modifies actual database state during execution   |
| Evaluation      | Deterministic database state verification         |
| Reproducibility | Exact state matching for consistent results       |

## License

The tau2-bench code is used under the MIT License. See the original repository for full license text.
