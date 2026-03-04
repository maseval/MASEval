# Available Benchmarks

This document provides detailed information, sources, and licensing for all benchmarks included in this library.

---

## 1. Multi-Agent Collaboration Scenario Benchmark (MACS Benchmark)

This benchmark is designed to test and evaluate the collaborative problem-solving capabilities of multi-agent systems. The implementation in this library provides the necessary code to set up and run these scenarios.

### Source and License

- **Original Repository:** [https://github.com/aws-samples/multiagent-collab-scenario-benchmark](https://github.com/aws-samples/multiagent-collab-scenario-benchmark)
- **Data License:** The dataset containing the scenarios is made available under the **Creative Commons Attribution 4.0 International License (CC-BY-4.0)**.

---

## 2. $\tau^2$-bench (Beta)

$\tau^2$-bench is a benchmark for evaluating agentic systems in realistic, multi-turn interactive environments.

> **Beta:** This benchmark has been implemented carefully, but it is highly complex and we have not yet validated the results against the original implementation. Use with caution when comparing with existing results or the original paper's numbers. Contributions and compute donations welcome!

### Source and License

- **Original Repository:** [https://github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- **Paper:** [Tau-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)
- **Code License:** MIT
- **Data License:** MIT

---

## 3. MultiAgentBench (MARBLE) (Beta)

MultiAgentBench is a comprehensive benchmark suite for evaluating multi-agent collaboration and competition in LLM-based systems. It includes diverse scenarios across multiple domains including research collaboration, negotiation, coding tasks, and more.

> **Beta:** This benchmark has been implemented carefully, but it is highly complex and we have not yet validated the results against the original implementation. Use with caution when comparing with existing results or the original paper's numbers. Contributions and compute donations welcome!

### Source and License

- **Original Repository:** [https://github.com/ulab-uiuc/MARBLE](https://github.com/ulab-uiuc/MARBLE) (where the original work was done)
- **Fork Used:** [https://github.com/cemde/MARBLE](https://github.com/cemde/MARBLE) (contains bug fixes for MASEval integration)
- **Paper:** [MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents](https://arxiv.org/abs/2503.01935)
- **Code License:** MIT
- **Data License:** MIT

> **Note**: MASEval uses a fork with bug fixes. All credit for the original work goes to the MARBLE team (Zhu et al., 2025).

---

## 4. GAIA2 (Beta)

Gaia2 is a benchmark for evaluating LLM-based agents on dynamic, multi-step scenarios using Meta's ARE (Agent Research Environments) platform. It tests agents across 7 capability dimensions: execution, search, adaptability, time, ambiguity, agent2agent, and noise.

> **Beta:** This benchmark has been implemented carefully, but it is highly complex and we have not yet validated the results against the original implementation. Use with caution when comparing with existing results or the original paper's numbers. Contributions and compute donations welcome!

### Source and License

- **Original Repository:** [https://github.com/facebookresearch/meta-agents-research-environments](https://github.com/facebookresearch/meta-agents-research-environments)
- **Paper:** [Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](https://openreview.net/forum?id=9gw03JpKK4) (ICLR 2026)
- **Dataset:** [https://huggingface.co/datasets/meta-agents-research-environments/gaia2](https://huggingface.co/datasets/meta-agents-research-environments/gaia2)
- **Code License:** MIT
- **Data License:** Subject to Meta's data usage terms (see HuggingFace dataset page)

---

## 5. CONVERSE (Beta)

CONVERSE evaluates contextual safety in agent-to-agent conversations. It focuses on adversarial interactions where an external service-provider agent attempts privacy extraction or unauthorized action induction over multiple turns.

> **Beta:** This benchmark has been implemented carefully, but it is highly complex and we have not yet validated the results against the original implementation. Use with caution when comparing with existing results or the original paper's numbers. Contributions and compute donations welcome!

### Source and License

- **Original Repository:** [https://github.com/amrgomaaelhady/ConVerse](https://github.com/amrgomaaelhady/ConVerse)
- **Paper:** [ConVerse: Contextual Safety in Agent-to-Agent Conversations](https://arxiv.org/abs/2506.15753)
- **Code License:** MIT (as provided by the upstream repository)
- **Data License:** Refer to the upstream repository's dataset and license terms

---

## 6. [Name of Next Benchmark]

(Description for the next benchmark...)

### Source and License

- **Original Repository:** [Link](Link)
- **Data License:** Data License.
