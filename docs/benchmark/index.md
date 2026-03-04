# Benchmarks

MASEval includes pre-implemented benchmarks for evaluating multi-agent systems.

!!! warning "Beta Benchmarks"
    Several benchmarks are currently in **Beta**. They have been implemented carefully, but these are highly complex systems and we have not yet validated the results against the original implementations. Use with caution when comparing with existing results or original paper numbers. Contributions and compute donations welcome!

    **MACS** is the only benchmark that has been fully validated.

## Adding Custom Benchmarks

You can also create your own benchmarks by subclassing the [`Benchmark`](../reference/benchmark.md) class. See the [Five-a-Day example](../examples/five_a_day_benchmark.ipynb) for a complete walkthrough.

## Licensing

For detailed source and licensing information for each benchmark's data, see [BENCHMARKS.md](https://github.com/parameterlab/MASEval/blob/main/BENCHMARKS.md).
