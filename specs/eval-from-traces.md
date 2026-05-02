# Spec: `Benchmark.eval()` — Re-evaluate from Existing Traces

**Status:** Draft
**Author:** (TBD)
**Target branch:** `eval-only-mode`

---

## 1. Motivation

Today, every evaluation requires a full execution: `setup_environment` → `setup_user` → `setup_agents` → `run_agents`/`execution_loop` → trace collection → `setup_evaluators` → `evaluate`. Steps 1–5 are by far the most expensive (LLM calls, tool simulators, sandboxed envs, network) and are not what changes when an author is iterating on evaluators.

Researchers iterating on evaluators (new metrics, prompt tweaks for LLM judges, scoring formulas, ablations) have no first-class way to re-run **only** the evaluation against previously captured runs. Workarounds today either rerun the whole benchmark (expensive, non-deterministic for non-seeded LLMs) or hand-stitch evaluators outside the framework (loses tracing, config, registry, callback machinery, and report schema).

We add `Benchmark.eval(reports, ...)` that takes previously-captured reports and produces fresh evaluation results, using **only** the data inside traces. Setup of environment/user/agents is **never** invoked.

## 2. Scope

In scope:

- New public method `Benchmark.eval(reports, tasks=None, agent_data=None, ...)`.
- Persisting `final_answer` in reports so it can round-trip.
- A breaking change to `setup_evaluators` to remove its dependency on live `environment`, `agents`, and `user` objects.
- Adapting in-tree benchmark evaluators (`mmlu`, `tau2`, `gaia2`, `macs`, `converse`, `multiagentbench`) to the new contract.
- Tests covering the round-trip (`run` → serialise → `eval`) and contract violations.
- Docs and a runnable example.

Out of scope:

- Storage formats beyond a passthrough Python list-of-dicts. Disk I/O is the caller's job (JSON, parquet, anything). We may later add `from_jsonl()` helpers, but not in this change.
- Re-execution / partial re-execution.
- Cross-benchmark trace replay (a tau2 trace cannot be evaluated by gaia2; we will validate task ids belong to the benchmark instance).

## 3. Guiding constraint (non-negotiable)

**Environments, users, and agents are not re-instantiated during `eval()`. Evaluators read all evaluation-relevant state from `traces` and from `task`.**

This implies:

- `Environment.gather_traces()` is the single source of truth for post-execution environment state used in evaluation. If a benchmark's evaluator currently reaches into a live env, the env's `gather_traces` must be extended to dump the equivalent data, and the evaluator refactored to read it from traces.
- The same applies to `User.gather_traces()` (user-side tool calls, internal scratchpad, termination reason).
- No "lazy re-instantiation" escape hatch in core. Benchmark authors who need ground-truth replay (e.g. tau2 `gold_environment`) build it from `task.environment_data` plus traces inside their evaluator — not from a live env handed to them by the framework.

## 4. Public API

### 4.1 `Benchmark.eval`

```python
def eval(
    self,
    reports: Sequence[Dict[str, Any]],
    *,
    tasks: Optional[Union[BaseTaskQueue, Iterable[Union[Task, dict]]]] = None,
    agent_data: Optional[Union[Dict[str, Any], Iterable[Dict[str, Any]]]] = None,
    seed: Optional[int] = None,
    seed_generator: Optional[SeedGenerator] = None,
) -> List[Dict[str, Any]]:
    """Re-evaluate previously-captured reports without re-running agents.

    For each input report, this method:
      1. Resolves the corresponding `Task`.
      2. Calls `setup_evaluators(task, traces, seed_generator)` (new signature, see §4.3).
      3. Calls `evaluate(evaluators, final_answer, traces)` (new signature, see §4.4).
      4. Returns a fresh report whose `traces`, `config`, `usage`, `final_answer`,
         and `task` fields are copied from the input, and whose `eval` and `status`
         reflect the replay.

    `eval()` never calls `setup_environment`, `setup_user`, `setup_agents`, or
    `run_agents`. It does not touch `execution_loop`. It is safe to call without
    network or model credentials, except for any judge models the evaluator itself
    creates.
    """
```

#### Parameters

- `reports`: Sequence of report dicts as produced by `run()`. Each must contain
  `task_id`, `repeat_idx`, `traces`, and `final_answer` (see §5 for the schema
  change). Reports with `status != "success"` are passed through unchanged
  (their `eval` stays `None`); we do not attempt to evaluate runs that didn't
  produce a complete trace.
- `tasks`: Optional task source. Required if the benchmark instance has no
  `self.tasks` from a prior `run()`, **or** if any report references a `task_id`
  not present in `self.tasks`. Accepts the same forms as `run(tasks=...)`.
- `agent_data`: Optional agent configuration. Carried into the seed generator's
  per-task scope only if the evaluator requests it (e.g., for LLM judges whose
  config you want to vary). Most evaluators ignore this. If omitted, an empty
  dict is used.
- `seed` / `seed_generator`: Same semantics as `Benchmark.__init__`, but **only**
  for evaluator seeds. Defaults to the benchmark's existing
  `self._seed_generator`. Re-evaluation is intentionally allowed to use a
  different seed than the original run (researchers comparing judge variance).

#### Returns

A list of report dicts with the same schema as `run()` (see §5).

#### Raises

- `ValueError` if a report references a `task_id` that cannot be resolved from
  `tasks` or `self.tasks`.
- `ValueError` if a successful report is missing `final_answer` or `traces`.
- `EvaluationError` (new exception, §7) if `setup_evaluators` or `evaluate`
  raises and `fail_on_evaluation_error=True`.

### 4.2 Behaviour around `fail_on_*`

`fail_on_setup_error` and `fail_on_task_error` are **ignored** in `eval()` —
neither stage runs. Only `fail_on_evaluation_error` applies, with the same
semantics as in `run()`.

### 4.3 Breaking change: `setup_evaluators` signature

Old (drops):

```python
def setup_evaluators(
    self,
    environment: Environment,
    task: Task,
    agents: Sequence[AgentAdapter],
    user: Optional[User],
    seed_generator: SeedGenerator,
) -> Sequence[Evaluator]: ...
```

New:

```python
def setup_evaluators(
    self,
    task: Task,
    traces: Dict[str, Any],
    seed_generator: SeedGenerator,
) -> Sequence[Evaluator]: ...
```

Rationale: forces evaluators to source state from traces. `agents` was already
unused in 4 of 6 in-tree benchmarks, and the remaining uses (tau2, gaia2) were
either (a) a dead reference (tau2 `self.environment` is stored but unused in
`__call__`; the actual gold-env replay reads from `self.task.environment_data`)
or (b) a true coupling (gaia2 reads live ARE judge state) that we are
deliberately breaking and migrating to a traces-based access path (§9.4).

`run()` passes `traces=self.collect_all_traces()` — i.e. the same traces the
evaluator will receive in `evaluate()`. This guarantees `setup_evaluators` and
`evaluate` see an identical view in both `run()` and `eval()` paths.

### 4.4 Breaking change: `evaluate` signature

Old:

```python
def evaluate(
    self,
    evaluators: Sequence[Evaluator],
    agents: Dict[str, AgentAdapter],
    final_answer: Any,
    traces: Dict[str, Any],
) -> List[Dict[str, Any]]: ...
```

New:

```python
def evaluate(
    self,
    evaluators: Sequence[Evaluator],
    final_answer: Any,
    traces: Dict[str, Any],
) -> List[Dict[str, Any]]: ...
```

`agents` is dropped. None of the in-tree benchmarks read it (verified — every
implementation has `_ = agents` or simply ignores the parameter). Agent state
that evaluators legitimately need is already part of `traces["agents"]`.

### 4.5 Updated `Evaluator.__init__`

The `Evaluator` ABC's `__init__` becomes:

```python
class Evaluator(ABC):
    def __init__(self, task: Task):
        self.task = task
```

`environment` and `user` are removed from `__init__`. Subclasses that need
state previously read from those objects must read it from `traces` in
`filter_traces` / `__call__`.

## 5. Report schema changes

Add one field, keep all existing fields:

```python
report = {
    "task_id": str,
    "repeat_idx": int,
    "status": str,                 # TaskExecutionStatus value
    "error": Optional[Dict],
    "traces": Dict,
    "config": Dict,
    "usage": Optional[Dict],
    "final_answer": Any,           # NEW — JSON-serialisable; None if not produced
    "eval": Optional[List[Dict]],
    "task": {"query": str, "metadata": Dict, "protocol": Dict},
}
```

`final_answer` is populated in `_execute_task_repetition` from the value
returned by `execution_loop`. Serialisability of `final_answer` is the
benchmark author's responsibility (most are strings/dicts; gaia2 returns a
structured dict, tau2/macs return strings or string lists).

`eval()` produces reports with the same schema. Replay-only fields:

- `status` is set to either `"success"` or `"evaluation_failed"`.
- `usage` from the input report is **carried through unchanged** for execution
  components, and **augmented** with any new usage from evaluator model calls
  made during replay (under a fresh `evaluators:*` registry key). This keeps
  per-component running totals accurate when the same `Benchmark` instance
  alternates `run()` and `eval()` calls.
- `config` is carried through, and the benchmark-level `config["benchmark"]`
  block is replaced with the *current* `gather_benchmark_config()` result,
  with the original captured under `config["benchmark_at_run_time"]`. This
  lets readers see both the original execution environment and the replay
  environment.
- `error` is `None` on success, populated with the evaluator's exception on
  failure.

## 6. Algorithm

```python
def eval(self, reports, *, tasks=None, agent_data=None, seed=None, seed_generator=None):
    # 1. Build/extend the task lookup.
    task_lookup = self._build_task_lookup_for_eval(tasks, reports)

    # 2. Resolve seed generator.
    eval_seed_gen = self._resolve_eval_seed_generator(seed, seed_generator)

    # 3. Resolve agent_data lookup (only used to thread into seed scope; default {}).
    agent_data_lookup = self._build_agent_data_lookup_for_eval(reports, agent_data)

    # 4. Clear reports list (mirrors run()).
    self.reports = []

    self._invoke_callbacks("on_run_start", self)
    for input_report in reports:
        self._invoke_callbacks("on_task_repeat_start", self, ...)
        out = self._evaluate_one_report(input_report, task_lookup, eval_seed_gen, agent_data_lookup)
        self._append_report_safe(out)
        self._invoke_callbacks("on_task_repeat_end", self, out)
    self._invoke_callbacks("on_run_end", self, self.reports)

    return self.reports
```

`_evaluate_one_report` is the single replay unit:

```python
def _evaluate_one_report(self, in_report, task_lookup, root_seed_gen, agent_data_lookup):
    task_id = in_report["task_id"]
    repeat_idx = in_report["repeat_idx"]
    task = task_lookup[task_id]                          # raises if missing
    traces = in_report.get("traces") or {}
    final_answer = in_report.get("final_answer")

    # Skip already-failed runs (no eval to perform).
    if in_report.get("status") != "success":
        return {**in_report, "eval": None}

    # Per-task scoped seed generator (mirrors _execute_task_repetition).
    task_seed_gen = root_seed_gen.for_task(task_id).for_repetition(repeat_idx)

    try:
        evaluators = self.setup_evaluators(task, traces, seed_generator=task_seed_gen)
        eval_results = self.evaluate(evaluators, final_answer, traces)
        new_status = TaskExecutionStatus.SUCCESS
        error_info = None
    except Exception as e:
        if self.fail_on_evaluation_error:
            self.clear_registry()
            raise
        eval_results = None
        new_status = TaskExecutionStatus.EVALUATION_FAILED
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        }

    # Collect any usage/configs/traces produced by replay-time evaluator components
    # (e.g. an LLM judge model adapter registered during setup_evaluators).
    replay_traces = self.collect_all_traces()
    replay_configs = self.collect_all_configs()
    replay_usage = self.collect_all_usage()
    self.clear_registry()

    return _build_replay_report(
        in_report=in_report,
        task=task,
        eval_results=eval_results,
        status=new_status,
        error=error_info,
        replay_traces=replay_traces,
        replay_configs=replay_configs,
        replay_usage=replay_usage,
    )
```

`_build_replay_report` rules (see §5):

- `traces` = `in_report["traces"]` (we do NOT merge replay traces — they go
  under `traces["evaluators_replay"]` to avoid colliding with execution-time
  agents/models traces).
- `config["benchmark"]` = current `gather_benchmark_config()`, original moved
  to `config["benchmark_at_run_time"]`.
- `usage` = element-wise sum of `in_report["usage"]` and `replay_usage`,
  preserving per-component breakdowns.

## 7. Errors and validation

- Add `EvaluationError(MASEvalError)` for failures inside `setup_evaluators` or
  `evaluate` during `eval()`. Distinct from `run()`'s in-line exceptions because
  here it is the only error class the caller can hit.
- Validate at `eval()` entry:
  - `reports` is non-empty.
  - Every successful report has `final_answer` and `traces` keys.
  - Every `task_id` resolves via `task_lookup`. Mismatched ids fail fast with a
    clear message ("report references task_id=X but only Y tasks were
    provided"). This catches the common error of mixing reports across
    benchmark instances.
- A per-report `_eval_replay_version` field is stamped (`"v1"`) so future
  schema migrations have a hook.

## 8. Concurrency

`eval()` is single-threaded in v1. Rationale:

- Replay is overwhelmingly dominated by LLM judge calls (where they exist),
  which are I/O-bound. Threading is plausible but introduces complexity with
  the registry's per-task clear/collect cycle.
- We can add `num_workers` support later (mirror `_run_parallel`) without API
  change.

`self._reports_lock` and `self._callback_lock` are still acquired so the path
is structurally compatible with future parallelism.

## 9. Migration of in-tree benchmarks

Each benchmark needs its `setup_evaluators`/`evaluate` rewritten to the new
signatures. Their evaluator implementations must source any environment/user
state from `traces` instead of stored `self.environment`/`self.user`.

### 9.1 mmlu — trivial

`MMLUEvaluator` stores `environment` but never uses it. Drop the parameter.

### 9.2 macs — trivial

`MACSEvaluator` does not depend on environment/user/agents. New
`setup_evaluators(task, traces, seed_generator)` is a one-line edit.

### 9.3 converse — trivial

Stores `environment`/`user` but does not invoke them in `__call__`. Drop the
parameters; reach for `task.environment_data` if domain info is needed.

### 9.4 tau2 — minor refactor

`Tau2Evaluator.__init__` stores `self.environment` but `_evaluate_environment`
already reconstructs envs via `get_environment_constructor(self.task.environment_data)`.
Drop the stored `self.environment`. No replay-blocking dependencies.

### 9.5 gaia2 — **this is the real work**

`Gaia2Evaluator.__call__` calls `self.environment.get_are_environment()` and
`self.environment.get_scenario()` at evaluation time, then mutates the
scenario's judge to drive `validate()`. The judge's accumulated turn-state is
populated *during the run* by the env, not exposed in traces.

Required migration:

1. `Gaia2Environment.gather_traces()` must serialise the post-execution state
   the judge needs. Concretely, this is the per-turn judge events / decisions
   produced by ARE during execution. ARE already snapshots these into the
   scenario; we serialise that snapshot.
2. `Gaia2Evaluator` is rewritten to construct an ARE judge from
   `task.evaluation_data["judge_engine_config"]` and replay it against the
   serialised state in `traces["environment"]["judge_state"]` plus
   `traces["agents"][...].messages`.
3. If the post-state cannot be fully serialised cheaply (likely for some ARE
   internals), the *minimal* invariant is: `gather_traces` captures whatever
   is needed for `validate()` to produce the same `gsr`, `partial_gsr`,
   `passed`, `rationale` it would have produced live. We add a contract test
   asserting that `eval(run_results)` produces evaluation outputs equal to the
   inline evaluation captured in the original report.

Acceptance criterion for gaia2: round-trip equality on at least one
representative scenario per ARE capability bucket.

### 9.6 multiagentbench — trivial

Does not use `environment`/`user`/`agents` in evaluator. Signature change only.

## 10. Tests

Add under `tests/test_core/test_benchmark_eval.py` (offline) and
`tests/test_benchmark/test_<bench>_eval.py` (per-benchmark, marked
`benchmark` + appropriate framework markers):

1. **Offline structural** (`core` marker):
   - A toy `Benchmark` subclass whose `setup_evaluators` returns a deterministic
     `Evaluator` reading `traces["x"]`. Assert `eval([report])` produces the
     same `eval` field as the original `run()` for a fabricated report.
   - `eval()` does not call `setup_environment`/`setup_user`/`setup_agents`/
     `run_agents` (assert via subclass that overrides each to raise).
   - Failed reports (`status != "success"`) pass through with `eval=None`.
   - Missing `final_answer` raises `ValueError` for a successful report.
   - Unknown `task_id` raises `ValueError` with a clear message.
   - `fail_on_evaluation_error=True` re-raises; `False` records
     `EVALUATION_FAILED` and continues.

2. **Round-trip equivalence** (`core` marker; deterministic evaluator only):
   - `reports = bench.run(tasks)` then `eval_reports = bench.eval(reports)` —
     `eval_reports[i]["eval"] == reports[i]["eval"]` for deterministic
     evaluators (mmlu, tau2 with no NL judge, multiagentbench rule-based).

3. **JSON round-trip** (`core` marker):
   - Serialise reports to JSON, deserialise, run `eval()`. Assert no failures
     and `eval` output matches.

4. **Per-benchmark live** (`live`+`credentialed`+`<framework>` markers):
   - One real run, persist reports, replay via `eval()`. Assert evaluation
     fields present and within tolerance for stochastic LLM-judge benchmarks
     (gaia2, tau2 with NL, macs, converse).

5. **gaia2 round-trip** (`live`+`gaia2`):
   - Specific test for §9.5 acceptance criterion.

6. **Concurrency invariants**: registry is empty after each
   `_evaluate_one_report` (assert `self._registry._is_empty()` between calls).

## 11. Docs

- New page `docs/guides/eval-from-traces.md` covering: when to use it, the
  contract that traces must carry all eval-relevant state, and the migration
  path for evaluators that previously reached into live env/user.
- Update `docs/getting-started/*` benchmark walk-through to mention
  `Benchmark.eval()` after the run example.
- Add a runnable `examples/eval_replay.py` that runs a small task set, dumps
  reports to a tempfile, and replays evaluation from disk.
- Update `AGENTS.md` "How to use" example for `Benchmark` to mention `eval()`.

## 12. Changelog

Under `Unreleased`:

- **Added** `Benchmark.eval()` to re-run evaluation from previously captured
  reports without re-executing agents. (PR: #PR_NUMBER_PLACEHOLDER)
- **Changed** `setup_evaluators` signature: `(task, traces, seed_generator)`.
  Environment, agents, and user are no longer passed in — read state from
  `traces`. (PR: #PR_NUMBER_PLACEHOLDER)
- **Changed** `evaluate` signature: `agents` parameter removed. (PR:
  #PR_NUMBER_PLACEHOLDER)
- **Changed** Reports now include `final_answer`. (PR: #PR_NUMBER_PLACEHOLDER)

## 13. Open questions

1. **Should `eval()` re-validate that `traces` is structurally well-formed
   before invoking `setup_evaluators`?** Proposal: no — the evaluator is the
   right layer to assert what it needs and produce a meaningful error. The
   framework only validates the report-level fields.
2. **Should we expose a CLI / `python -m maseval.eval reports.jsonl`?** Out of
   scope for this PR; trivial follow-up once `eval()` lands.
3. **Should `final_answer` be stripped or hashed for size control?** Some
   benchmarks (gaia2) return large structured dicts. Decision: store as-is; if
   size becomes a problem, callers can post-process before persisting. We do
   not invent serialisation policies in core.
4. **Should `eval()` accept a single report or always a sequence?** Always a
   sequence; callers wrap a single report in `[r]`. Mirrors `run(tasks=...)`'s
   normalisation pattern.

## 14. Implementation order

1. Add `final_answer` to report schema in `_execute_task_repetition`. Land
   alone — trivially additive, no breaking change.
2. Migrate `setup_evaluators`/`evaluate` signatures and update all in-tree
   benchmarks except gaia2. Land as one PR (sweeping, but mechanical).
3. Migrate gaia2 evaluator + env trace serialisation. Separate PR with its
   own round-trip test.
4. Implement `Benchmark.eval()` and tests. Land last so steps 1–3 are
   already exercised by `run()` before adding the replay path.

This staging means we never have a half-broken main branch: each PR independently passes the full test suite.
