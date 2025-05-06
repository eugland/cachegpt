# CacheDistill: Energy-Efficient Query Routing for Large Language Models via Edge Caching and Semantic Similarity

## Abstract

Large language models (LLMs) are powerful but come with substantial energy and cost burdens, particularly when kept in active memory for real-time inference. While storing models on secondary storage is inexpensive, the energy required to keep them memory-resident and responsive is non-trivial.

This paper proposes a sustainable, hybrid edge-cloud inference architecture that reduces energy consumption by delegating routine or high-confidence queries to lightweight, distilled models running at the edge, while reserving cloud-based LLMs for only the most complex tasks.

Beyond traditional confidence-based delegation, we introduce a semantic caching mechanism inspired by heuristics from internet-scale systems such as search engines and video platforms. By clustering and caching semantically similar user queries, the system minimizes redundant inference and incrementally strengthens edge performance.

This design aligns with the conference’s emphasis on sustainable AI: it not only reduces emissions associated with always-on cloud inference but also offers a scalable and energy-efficient approach to LLM deployment in practice.

---

## Introduction

The rapid advancement of large language models (LLMs) has unlocked unprecedented capabilities in natural language understanding and generation. However, these gains come at a steep cost: deploying LLMs in real-time applications requires keeping them loaded in memory on high-performance GPUs or TPUs, leading to significant energy consumption and operational expense.

This constraint poses a critical barrier to sustainable AI deployment, especially as demand for low-latency, ubiquitous access to intelligent systems grows.

While cloud-based inference offers scalability, it is energy-intensive and poorly optimized for handling simple or frequently repeated queries. Recent approaches have explored offloading tasks to smaller, distilled models at the edge, but most rely solely on confidence thresholds for delegation—limiting their adaptability and failing to leverage the structure in repeated user behavior.

In this work, we propose a novel edge-cloud LLM architecture that improves sustainability without compromising performance. At the core of our system is a semantic caching and query distillation mechanism, inspired by heuristics used in large-scale content recommendation and search platforms.

When a user query is submitted, it is first handled by a lightweight edge model. The system compares the query against a growing cache of past questions and clusters them based on semantic similarity. If a match is found or confidence is sufficiently high, the edge model returns the result; otherwise, the query is escalated to a more capable cloud-hosted model.

This architecture reduces redundant computation, minimizes unnecessary cloud inference, and continually adapts to user behavior. By combining caching, query distillation, and selective delegation, we present a scalable and energy-efficient framework aligned with the goals of sustainable AI system design.

---

## Background

### State of the Art Inference Cost

| Model               | Params (B) | GPU VRAM | CPU RAM | Power (W) | Inference Time (s) | Energy (Wh) | Cost per 1K Tokens |
|---------------------|------------|----------|---------|-----------|---------------------|--------------|---------------------|
| LLaMA 7B            | 7          | 16 GB    | 32 GB   | 100       | 5                   | 0.14         | $0.014              |
| LLaMA 13B           | 13         | 24 GB    | 64 GB   | 150       | 6                   | 0.25         | $0.025              |
| LLaMA 30B           | 30         | 48 GB    | 128 GB  | 250       | 8                   | 0.56         | $0.056              |
| LLaMA 65B           | 65         | 80 GB    | 256 GB  | 400       | 10                  | 1.11         | $0.111              |
| GPT-J 6B            | 6          | 16 GB    | 32 GB   | 90        | 5                   | 0.125        | $0.013              |
| GPT-NeoX 20B        | 20         | 40 GB    | 96 GB   | 200       | 7                   | 0.39         | $0.039              |
| DeepSeek-R1 7B      | 7          | 16 GB    | 32 GB   | 100       | 5                   | 0.14         | $0.014              |
| DeepSeek-R1 32B     | 32         | 48 GB    | 128 GB  | 250       | 8                   | 0.56         | $0.056              |
| Qwen 7B             | 7          | 16 GB    | 32 GB   | 90        | 5                   | 0.125        | $0.013              |
| Qwen 14B            | 14         | 24 GB    | 64 GB   | 150       | 6                   | 0.25         | $0.025              |
| Qwen 72B            | 72         | 80 GB    | 256 GB  | 400       | 10                  | 1.11         | $0.111              |
| Ollama (LLaMA 7B)   | 7          | 16 GB    | 32 GB   | 100       | 5                   | 0.14         | $0.014              |
| Ollama (LLaMA 13B)  | 13         | 24 GB    | 64 GB   | 150       | 6                   | 0.25         | $0.025              |

As shown in the table, approximately 90% of a model's energy consumption occurs during inference. This is because training is a one-time process, while inference happens continuously as users around the world interact with the model.

The data also clearly indicates that larger models are more power-intensive. Conversely, smaller models incur significantly lower energy costs. This suggests an intuitive strategy: **favor smaller models for efficiency**.

However, smaller models often struggle to match the performance of larger ones, as demonstrated in various studies *(enumerate studies here)*. The challenge, then, is to retain the capabilities of large language models (LLMs) while reducing the computational cost.

To address this, we propose a hybrid solution that combines semantic caching with a distilled edge model.
