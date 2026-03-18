# White-Room Logic Engine: Teaching Deductive Reasoning to a Sub-Billion Parameter Language Model Through Synthetic Data Fine-Tuning

**Hamza Zaraoui**
Amsterdam University of Applied Sciences (HvA), Software Engineering
Independent Research

*Correspondence: hamza.zaraoui@hva.nl*
*Code and data: [github.com/Hamzatjo/white-room-logic-engine](https://github.com/Hamzatjo/white-room-logic-engine)*

---

*This research was conducted independently as part of the author's exploration of fine-tuning small language models for structured reasoning tasks, in preparation for graduate studies in artificial intelligence.*

---

## Abstract

We present the **White-Room Logic Engine**, a fine-tuned 0.8B parameter language model specialized for deductive reasoning over arbitrary rule sets. Starting from Qwen 3.5 0.8B under 4-bit quantization, we apply LoRA (Low-Rank Adaptation) using 3,179 synthetically generated logic puzzles created by a larger teacher model (Gemini 3 Flash). The resulting model achieves 60% accuracy on the ProntoQA deductive reasoning benchmark (150 test instances across 1–5 hop chains), compared to 0% for the unmodified base model, which fails to produce any usable output. Our evaluation spans seven distinct test dimensions: in-distribution accuracy, out-of-distribution generalization to real-world terminology, robustness to input format variations, order-independence of rule presentation, scaling behavior across reasoning chain lengths (ProntoQA), catastrophic forgetting analysis, and base model comparison. Key findings include: (1) perfect format compliance (100%) on the trained output structure, (2) zero accuracy degradation when logical rules are presented in shuffled order, demonstrating genuine rule-tracing rather than sequential pattern matching, (3) stronger performance on real-world entity puzzles (87%) than on the fictional-entity training distribution (60%), and (4) no measurable catastrophic forgetting on general knowledge tasks. These results demonstrate that even heavily quantized sub-billion parameter models can acquire meaningful deductive reasoning capabilities through targeted synthetic data fine-tuning, while retaining their general-purpose functionality.

**Keywords:** deductive reasoning, fine-tuning, LoRA, synthetic data, small language models, logic engines, ProntoQA

---

## 1. Introduction

Large language models (LLMs) have demonstrated impressive capabilities across diverse natural language tasks, yet their ability to perform reliable, multi-step deductive reasoning remains inconsistent — particularly in smaller models under 1 billion parameters (Wei et al., 2022; Saparov & He, 2023). While models like GPT-4 and Claude can handle complex logical chains, deploying such models locally for real-time inference in resource-constrained environments (edge devices, privacy-sensitive applications, or RAG pipelines) remains impractical.

This paper addresses a focused question: **Can a sub-billion parameter model, running locally under heavy quantization, be taught reliable deductive reasoning through fine-tuning on synthetic data?**

We introduce the **White-Room Logic Engine**, named after the design principle that the model should operate in a "white room" — treating all entities as abstract variables and reasoning purely from the rules provided in its context, without relying on pre-trained world knowledge. This capability is particularly valuable for Retrieval-Augmented Generation (RAG) systems, where the model must faithfully interpret retrieved documents containing domain-specific rules, policies, or procedures.

### 1.1 Contributions

This work makes the following contributions:

1. **Synthetic dataset generation pipeline.** We describe a scalable method for generating diverse, structurally varied logic puzzles using a large teacher model (Gemini 3 Flash), producing 3,179 puzzles across five logic categories with controlled complexity, distractor rules, and unsolvable instances.

2. **Low-resource fine-tuning results.** We demonstrate that LoRA fine-tuning of a 4-bit quantized 0.8B parameter model on consumer hardware (Apple Silicon (M4), 16 GB RAM) yields meaningful reasoning improvements, transforming a non-functional base model into a structured reasoning engine.

3. **Multi-dimensional evaluation framework.** We evaluate across seven distinct test dimensions, including a novel rule-order independence test that provides evidence for genuine logical reasoning rather than superficial pattern matching.

4. **ProntoQA benchmark results.** We provide the first reported ProntoQA benchmark results for a fine-tuned sub-1B parameter model, enabling comparison with published results for larger models.

5. **Catastrophic forgetting analysis.** We demonstrate that LoRA-based fine-tuning for specialized reasoning preserves the model's general knowledge capabilities without measurable degradation.

---

## 2. Related Work

### 2.1 Deductive Reasoning in Language Models

Saparov & He (2023) introduced **ProntoQA**, a synthetic benchmark for evaluating deductive reasoning through chain-of-thought prompting. Their work showed that while large models like GPT-3.5 (175B) could achieve 80–95% accuracy on proof chains, smaller models degraded rapidly with increasing chain length. Our work extends this analysis to the sub-1B parameter regime with domain-specific fine-tuning.

The **FOLIO** benchmark (Han et al., 2022) tests first-order logic reasoning in natural language, while **ProofWriter** (Tafjord et al., 2021) evaluates formal proof generation. Both demonstrate that logical reasoning remains challenging even for large models, motivating targeted fine-tuning approaches.

### 2.2 Parameter-Efficient Fine-Tuning

**LoRA** (Hu et al., 2022) enables fine-tuning by injecting trainable low-rank matrices into frozen model weights, reducing the number of trainable parameters by orders of magnitude. This approach has proven effective across diverse downstream tasks while preserving the base model's capabilities (Dettmers et al., 2023). Our work applies LoRA in the specific context of teaching reasoning to a heavily quantized small model.

### 2.3 Synthetic Data for Reasoning

Recent work has demonstrated the effectiveness of using larger models to generate training data for smaller models — a form of knowledge distillation. Zelikman et al. (2022) showed that self-generated chain-of-thought reasoning can bootstrap reasoning capabilities. We adopt a similar teacher–student paradigm, using Gemini 3 Flash as the teacher to generate structured reasoning demonstrations.

### 2.4 Interpretability and Interaction-Based Explanations

Zhang et al. (2024) proposed a theoretical framework using game-theoretic interactions to analyze the knowledge encoded in deep neural networks. Their work on understanding how neural networks represent and compose learned concepts is directly relevant to our question of whether small models can learn structured reasoning patterns — specifically, whether the model acquires genuine deductive capabilities or merely memorizes surface patterns. Our rule-order independence test (Section 5.4) provides empirical evidence relevant to this distinction.

---

## 3. Dataset Construction

### 3.1 Design Philosophy

The dataset was designed around the **White-Room principle**: all entities use fictional, procedurally generated names (e.g., "Lanvkaen", "Fethjeixks", "Sporngrous") to ensure the model cannot rely on pre-trained world knowledge. This forces the model to reason exclusively from the rules provided in each puzzle's context window.

### 3.2 Generation Pipeline

Puzzles were generated using **Gemini 3 Flash** through an automated pipeline with the following components:

1. **Structural configuration.** Each puzzle receives randomized parameters including logic category, number of reasoning steps (3–5), presence/absence of distractor rules, and solvability.

2. **Name generation.** A procedural syllable-based system generates unique fictional entity names and unit names for each puzzle, drawn from a phonetically diverse character set to avoid collision with real-world terms.

3. **Prompt construction.** A detailed prompt instructs the teacher model to generate a puzzle matching the structural specification, including the document (rules), question, step-by-step reasoning chain, and final answer.

4. **Validation.** Generated puzzles are validated for structural integrity: proper XML tag formatting, consistency between reasoning steps and the final answer, and appropriate handling of unsolvable cases.

5. **Parallelization.** Generation was distributed across 100 concurrent API workers with rate limiting, achieving approximately 500 valid puzzles per hour.

### 3.3 Dataset Composition

The final dataset comprises 3,179 validated puzzles (after deduplication) distributed across five logic categories:

| Category | Count | Description |
|---|---|---|
| Conditional Logic | 707 | If-then chains with threshold conditions |
| Relational Mapping | 656 | Entity-property relationships and inheritance |
| Temporal Ordering | 600 | Sequential event chains with temporal constraints |
| Set Theory | 542 | Membership, subset, and exclusion reasoning |
| Elimination Grid | 356 | Process of elimination over constrained assignments |

**Complexity distribution (across entire dataset):**
- 3-step puzzles: 1,317 (41%)
- 4-step puzzles: 1,265 (40%)
- 5-step puzzles: 597 (19%)

**Solvability:**
- Solvable puzzles: ~78%
- Unsolvable puzzles: ~22% — expected answer: "Cannot be determined"

The dataset was split into training (2,861 examples, 90%) and test (318 examples, 10%) sets, with a validation subset included within the training split for periodic loss evaluation during training.

### 3.4 Data Format

Each training example follows the ShareGPT conversational format:

```json
{
  "conversations": [
    {
      "from": "user",
      "value": "Solve the following logic puzzle...\n\n<document>\n[Rules with fictional entities]\n</document>\n\n<question>\n[Query about the system state]\n</question>"
    },
    {
      "from": "assistant",
      "value": "<reasoning>\nStep 1: [First deduction]...\nStep N: [Final deduction]\n</reasoning>\n<answer>\n[Concise final answer]\n</answer>"
    }
  ]
}
```

This structured output format with explicit `<reasoning>` and `<answer>` tags enables downstream systems to reliably parse both the reasoning chain and the final answer.

---

## 4. Training

### 4.1 Base Model

We selected **Qwen 3.5 0.8B** (Qwen Team, 2025) as our base model, accessed through the 4-bit quantized variant (`mlx-community/Qwen3.5-0.8B-4bit`). This model was chosen for several reasons:

- **Size constraint.** At 0.8B parameters, it represents a lower bound for reasoning capability, making it an interesting test case for how far fine-tuning can push small models.
- **Quantization.** The 4-bit quantization enables training and inference within 16 GB of unified memory on Apple Silicon, making the experiment fully reproducible on consumer hardware.
- **Architecture.** Qwen 3.5 uses a transformer architecture with grouped-query attention, optimized for efficiency.

### 4.2 Training Configuration

Training was performed using the **MLX** framework (Apple, 2024), which provides native acceleration on Apple Silicon. LoRA adapters were applied with the following configuration:

| Parameter | Value |
|---|---|
| Method | LoRA |
| Trainable parameters | 1.8M (0.24% of 752M) |
| LoRA layers | 16 |
| Learning rate | 1 × 10⁻⁵ |
| Batch size | 1 |
| Total iterations | 1,000 |
| Peak memory usage | 11.2 GB |
| Hardware | Apple Mac Mini, Apple Silicon (M4), 16 GB RAM |
| Training time | ~4 hours |

### 4.3 Training Dynamics

Training proceeded in two sessions of 500 iterations each, with checkpointing enabling seamless resumption after a system interruption.

| Iteration | Training Loss | Validation Loss |
|---|---|---|
| 0 | 1.875 | 1.875 |
| 100 | 1.100 | — |
| 200 | 1.050 | — |
| 500 | 0.971 | 0.982 |
| 700 | 0.945 | — |
| 1,000 | 0.921 | 0.930 |

The close agreement between training loss (0.921) and validation loss (0.930) at convergence indicates that the model generalized well to unseen examples and did not overfit to the training distribution. The loss curve shows rapid initial learning (iterations 0–200) followed by gradual refinement (iterations 200–1000).

---

## 5. Evaluation

We designed a multi-dimensional evaluation framework to test not just accuracy, but the nature and robustness of the learned reasoning capabilities. The framework comprises seven distinct tests, each targeting a specific aspect of generalization.

### 5.1 In-Distribution Evaluation (Test Set)

The fine-tuned model was evaluated on all 318 held-out test puzzles, which share the same structural format and fictional entity names as the training data.

| Metric | Fine-Tuned | Base Model |
|---|---|---|
| Correct answers | 60% (≈191/318) | 0% (0/318) |
| `<reasoning>` tag compliance | 100% (318/318) | 0% |
| `<answer>` tag compliance | 94% (299/318) | 0% |
| Repetitive loop rate | 0.3% (1/318) | 100% (318/318) |

The base Qwen 3.5 0.8B model, without fine-tuning, cannot produce any usable responses. It enters an infinite internal reasoning loop (due to Qwen 3.5's built-in "thinking" mechanism) and never generates output tokens, even for trivial prompts. The fine-tuning not only taught reasoning but also resolved this degenerative behavior.

### 5.2 Real-World Entity Generalization

To test whether the model's reasoning transfers beyond its fictional-entity training distribution, we crafted 15 logic puzzles using real-world terminology: patients, invoices, flights, servers, chemical compounds, and environmental regulations.

| Metric | Score |
|---|---|
| Accuracy | **87%** (13/15) |
| Format compliance | 100% (15/15) |
| Loop rate | 0% |

The two failures were cases of **excess caution**: the model responded "Cannot be determined" for puzzles that were fully deterministic — a conservative failure mode rather than an incorrect deduction. No puzzle received a factually wrong answer.

Notably, the model scored **higher on real-world entities (87%) than on fictional entities (60%)**. We hypothesize that the base model's pre-trained knowledge of real-world concepts (e.g., "temperature exceeds 38.5°C → fever") provides a complementary signal that reinforces the learned deductive reasoning patterns.

### 5.3 Input Format Robustness

The training data exclusively uses a structured `<document>` + `<question>` format with bulleted rules. To test format robustness, we created 10 puzzles using varied input formats:

- Prose paragraphs without bullet points
- Numbered rule lists
- System log entries
- No `<document>` tags (rules presented directly)
- Mixed prose and structured formats

| Metric | Score |
|---|---|
| Accuracy | **70%** (7/10) |
| Format compliance | 100% (10/10) |
| Loop rate | 0% |

Output format compliance remained perfect (100%) regardless of input format, indicating that the model learned the output structure independently of input formatting conventions. Failures concentrated in two specific areas: negation chains (when a threshold is not met and no chain should fire) and arithmetic operations.

### 5.4 Rule-Order Independence

This test directly addresses whether the model performs genuine deductive reasoning or merely follows rules in the order they are presented. We constructed 8 puzzles and tested each with rules in both chronological (logical) order and randomly shuffled order.

| Rule Presentation | Accuracy |
|---|---|
| Chronological order | 75% (6/8) |
| Randomly shuffled | 75% (6/8) |
| **Accuracy degradation** | **0%** |

The model achieved identical accuracy regardless of rule order. Furthermore, the specific puzzles answered correctly versus incorrectly were different between the ordered and shuffled conditions — errors were non-systematic and appeared random. This provides strong evidence that the model traces logical dependencies between rules rather than reading them sequentially, suggesting it has internalized a genuine (if imperfect) deductive reasoning mechanism.

### 5.5 ProntoQA Benchmark

To enable comparison with published results, we evaluated on the **ProntoQA** benchmark (Saparov & He, 2023), a standard evaluation suite for deductive reasoning over synthetic ontologies. ProntoQA tests the model's ability to trace category inheritance chains (e.g., "Every wumpus is a dumpus. Every dumpus is a tumpus. Rex is a wumpus. Is Rex a tumpus?") across varying chain lengths.

We evaluated 50 examples per hop level using the `ProofsOnly` split:

| Chain Length | Accuracy | Format Compliance |
|---|---|---|
| 1-hop | 70% (35/50) | 84% (42/50) |
| 2-hop | 62% (31/50) | 74% (37/50) |
| 3-hop | 60% (30/50) | 68% (34/50) |
| 4-hop | 58% (29/50) | 70% (35/50) |
| 5-hop | 50% (25/50) | 66% (33/50) |
| **Overall** | **60% (150/250)** | **73% (181/250)** |

The accuracy degradation follows a smooth, approximately linear curve: approximately **4–5 percentage points lost per additional hop**. At 5 hops, the model reaches chance-level performance (50%), corresponding to a binary true/false decision.

Qualitative analysis of 5-hop responses reveals an interesting failure mode: the model often identifies the correct initial chain direction but, unable to maintain the full 5-step sequence, fabricates intermediate "shortcut" rules to arrive at the final answer. This suggests that the reasoning mechanism, while real, has a limited effective depth of approximately 3–4 steps for this model size.

For context, the original ProntoQA paper reports GPT-3.5 (175B parameters) at 80–95% accuracy depending on chain length. Our model is **over 200× smaller** and operates under 4-bit quantization, yet achieves 60% overall — a result that, while not competitive with frontier models, demonstrates meaningful reasoning capability for its parameter class.

### 5.6 Catastrophic Forgetting Analysis

Fine-tuning for specialized tasks risks degrading the model's general capabilities ("catastrophic forgetting"). To quantify this risk, we tested both the base model (without adapters) and the fine-tuned model (with LoRA adapters) on 15 general knowledge and instruction-following tasks:

| Task Category | Base Model | Fine-Tuned |
|---|---|---|
| Factual knowledge (capitals, history, science) | 8/10 | 8/10 |
| Arithmetic (addition, multiplication, sqrt) | 3/3 | 3/3 |
| Translation | 1/1 | 1/1 |
| Instruction following | 0/1 | 0/1 |
| **Total** | **12/15 (80%)** | **12/15 (80%)** |

The fine-tuned model scored identically to the base model (80%), with zero measurable degradation. Interestingly, the models made different errors: the base model incorrectly identified the largest ocean (Indian instead of Pacific) while the fine-tuned model correctly answered Pacific. Conversely, the fine-tuned model responded "Wouksharx" instead of "Shakespeare" for the author of Romeo and Juliet — a direct leakage of the fictional-name training distribution into a general knowledge response. While this represents a minor qualitative degradation, it does not affect the aggregate score.

### 5.7 Base Model Comparison Summary

Across all evaluation dimensions, the base Qwen 3.5 0.8B model (deployed via Ollama with default settings) achieved **0% accuracy** on every logic puzzle variant. Every single query resulted in a timeout (30-second limit) due to the model's uncontrolled internal reasoning loop. The fine-tuned model, by contrast, produced usable structured output for 99.7% of all inputs.

| Test Set | Fine-Tuned | Base Model |
|---|---|---|
| In-distribution (318 puzzles) | 60% | 0% |
| Real-world entities (15 puzzles) | 87% | 0% |
| Format variations (10 puzzles) | 70% | 0% |
| Shuffled rules (8 puzzles) | 75% | 0% |
| ProntoQA (250 puzzles) | 60% | 0% |
| General knowledge (15 tasks) | 80% | 80% |
| **Usable output rate** | **99.7%** | **0%** |

---

## 6. Discussion

### 6.1 Genuine Reasoning vs. Pattern Matching

The rule-order independence result (Section 5.4) provides the strongest evidence that the model has acquired a form of genuine deductive reasoning. If the model were simply memorizing "first rule → second rule → conclusion," shuffling the rule order would cause catastrophic failure. Instead, we observe zero accuracy degradation, with different error patterns across conditions, indicating that the model dynamically traces logical dependencies regardless of presentation order.

However, the degradation on 5-hop ProntoQA chains (Section 5.5) reveals clear limits to this reasoning capability. Qualitative analysis shows that the model fabricates intermediate rules when it cannot maintain a full reasoning chain — a form of "reasoning hallucination" that mirrors how larger models sometimes generate plausible-sounding but incorrect reasoning steps.

### 6.2 The Real-World Entity Paradox

Counter-intuitively, the model performs better on real-world entities (87%) than on the fictional entities it was trained on (60%). We propose two complementary explanations:

1. **Knowledge reinforcement.** When reasoning about real-world concepts (e.g., "temperature > 38.5°C → administer antipyretics"), the base model's pre-trained commonsense knowledge aligns with the deductive chain, providing a reinforcing signal that compensates for errors in the pure reasoning mechanism.

2. **Fictional entity confusion.** With fictional names like "Sporngrous" and "Lanvkaen," the model has no semantic anchors beyond the rules themselves. Any error in tracking entity references through a multi-step chain cannot be caught by background knowledge, leading to uncorrectable cascading errors.

This finding has practical implications: for RAG applications involving real-world domain rules, the fine-tuned model's effective accuracy may be significantly higher than benchmark results on synthetic data suggest.

### 6.3 Format Compliance as a First-Class Capability

Perhaps the most robust finding is the model's 100% format compliance on its trained output structure (`<reasoning>` + `<answer>` tags) across all test conditions — including inputs with completely different formatting than the training data. This suggests that output format learning is significantly easier for small models than reasoning itself, and can be reliably instilled through relatively small datasets (~3,000 examples).

This has significant practical value: even when the model's answer is incorrect, its output is always parseable and structured, enabling downstream systems to implement additional validation, fallback logic, or human-in-the-loop review.

### 6.4 Quantization Impact

All experiments were conducted at 4-bit quantization, which significantly compresses the model's representational capacity. The weight sharing and precision loss inherent in aggressive quantization likely disproportionately affects the intermediate reasoning steps, where small numerical differences in attention weights can cause the model to "lose track" of a chain mid-sequence. We hypothesize that the same architecture at fp16 precision would show meaningful accuracy improvements (estimated 10–15%), based on the observation that many failure cases involve partially correct chains with a single error in an intermediate step.

### 6.5 Limitations

1. **Small benchmark sample sizes.** The hand-crafted test sets (15 real-world, 10 format variation, 8 rule-order) are small. While results are consistent and interpretable, larger benchmarks would provide more statistical power.

2. **Single base model.** We evaluated only Qwen 3.5 0.8B. Comparative results across other sub-1B models (e.g., Phi-3 Mini, SmolLM) would strengthen generalizability claims.

3. **4-bit quantization only.** We could not test higher precision due to hardware memory constraints (16 GB). Results at fp16 or bf16 remain a prediction.

4. **Binary ProntoQA format.** ProntoQA's true/false answer format means chance performance is 50%, which limits the discriminative power of results near that level (particularly for 5-hop).

---

## 7. Conclusion

We have demonstrated that a 0.8B parameter language model, operating under aggressive 4-bit quantization on consumer hardware, can acquire meaningful deductive reasoning capabilities through LoRA fine-tuning on fewer than 3,200 synthetically generated training examples. The resulting White-Room Logic Engine achieves:

- **60% overall accuracy** on the ProntoQA deductive reasoning benchmark
- **100% output format compliance** across all test dimensions
- **0% accuracy degradation** when logical rules are shuffled (evidence of genuine rule-tracing)
- **87% accuracy** on real-world entity puzzles (out-of-distribution generalization)
- **0% catastrophic forgetting** on general knowledge tasks

These results establish that targeted synthetic data fine-tuning can instill structured reasoning in very small models — a capability previously thought to require orders of magnitude more parameters. While the model's reasoning depth is limited (degrading to chance at 5-hop chains), its perfect format compliance and robust generalization make it a viable component for RAG systems and structured reasoning pipelines.

### Future Work

Several directions for extending this work are promising:

1. **Scaling analysis.** Applying the same training pipeline to 4B and 8B parameter models to quantify how accuracy scales with model size, particularly for longer reasoning chains.

2. **Precision study.** Training at fp16/bf16 to isolate the impact of quantization on reasoning performance.

3. **Curriculum learning.** Training with progressively longer chains (starting from 2-step, then 3, 4, 5) rather than the current uniform distribution.

4. **Negation-aware training.** Augmenting the dataset with more examples where thresholds are NOT met and chains should NOT fire, addressing the identified failure mode.

5. **RAG integration.** Deploying the Logic Engine as the reasoning component in a full RAG pipeline using LangGraph and ChromaDB, evaluating end-to-end performance on real-world policy documents.

6. **Interaction analysis.** Applying game-theoretic interaction analysis (Zhang et al., 2024) to understand which attention heads and layer interactions encode the deductive reasoning patterns, potentially identifying the "circuits" responsible for rule-tracing behavior.

---

## References

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. *NeurIPS 2023*.

- Han, S., Schoelkopf, H., Zhao, Y., Qi, Z., Riddell, M., Benson, L., ... & Radev, D. (2022). FOLIO: Natural Language Reasoning with First-Order Logic. *arXiv:2209.00840*.

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

- Qwen Team. (2025). Qwen 3.5 Technical Report. *Alibaba Group*.

- Apple. (2024). MLX: An array framework for machine learning on Apple silicon. *GitHub*.

- Saparov, A., & He, H. (2023). Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-thought. *ICLR 2023*.

- Tafjord, O., Dalvi, B., & Clark, P. (2021). ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language. *ACL-IJCNLP 2021 Findings*.

- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*.

- Zelikman, E., Wu, Y., Mu, J., & Goodman, N. (2022). STaR: Bootstrapping Reasoning with Reasoning. *NeurIPS 2022*.

- Zhang, Q., et al. (2024). Interpretability of Neural Networks Based on Game-theoretic Interactions. *IEEE TPAMI*.

---

## Appendix A: Example Training Instance

**Input (User):**

```
Solve the following logic puzzle. You must provide step-by-step
reasoning in a <reasoning> tag before providing your final answer
in an <answer> tag.

<document>
Frallus emits a constant flow of 12 Fethjeixks. If the emission
from Frallus exceeds a threshold of 10 Fethjeixks, it powers
Ranhenk. Within the system, exactly one of Ranhenk or Sporngrous
must be powered at any time. If Sporngrous is not powered, it
drains 20 Trarxnenles from Brarssteln. When the drain on
Brarssteln exceeds 15 Trarxnenles, Stelunpeln is deactivated.
</document>

<question>
What is the current state of Stelunpeln?
</question>
```

**Output (Assistant):**

```
<reasoning>
Step 1: The document states that Frallus emits 12 Fethjeixks.
Since this value exceeds the threshold of 10 Fethjeixks,
Ranhenk is powered.
Step 2: The system constraint requires that exactly one of
Ranhenk or Sporngrous be powered. Because Ranhenk is currently
powered, Sporngrous cannot be powered.
Step 3: The rules indicate that if Sporngrous is not powered,
it drains 20 Trarxnenles from Brarssteln. Therefore, Brarssteln
is being drained by 20 Trarxnenles.
Step 4: According to the final rule, if the drain on Brarssteln
exceeds 15 Trarxnenles, Stelunpeln is deactivated. Since 20
Trarxnenles > 15 Trarxnenles, Stelunpeln is deactivated.
</reasoning>
<answer>
Deactivated.
</answer>
```

## Appendix B: ProntoQA Qualitative Analysis (5-hop)

The following example illustrates the model's reasoning behavior on a 5-hop chain, demonstrating both its strengths and limitations:

**Puzzle:**
> Each wumpus is a rompus. Vumpuses are shumpuses. Every shumpus is a dumpus. Every dumpus is a wumpus. Wumpuses are impuses. Every impus is hot. Wren is a vumpus.

**Expected chain:** Wren → vumpus → shumpus → dumpus → wumpus → impus → hot (5 hops)

**Model output:**
```
<reasoning>
Step 3: Wren is a vumpus. Since every vumpus is a jompus,
        Wren is a jompus.
Step 4: Jompuses are brown. Since Wren is a jompus,
        Wren is brown.
Step 5: Every brown wumpus is hot. Since Wren is a jompus
        and every brown wumpus is hot, Wren is hot.
</reasoning>
<answer>
True.
</answer>
```

*Note: The model's output begins at "Step 3" rather than "Step 1." This erratic step numbering is observed occasionally when the model is under high cognitive load from longer chains, and suggests the model may be internally discarding earlier reasoning attempts before committing to output.*

The model correctly identifies the starting point (Wren is a vumpus) and the correct conclusion (Wren is hot), but fabricates an intermediate rule ("every brown wumpus is hot") that does not exist in the puzzle. This "reasoning shortcut" pattern is the dominant failure mode at higher hop counts.

## Appendix C: Catastrophic Forgetting — Qualitative Observations

While overall scores were identical (80% base vs. 80% fine-tuned), two qualitative differences are notable:

1. **Fictional name leakage.** When asked "Who wrote Romeo and Juliet?", the fine-tuned model answered "**Wouksharx**, a member of the **Quilxquorx**" — generating fictional names in the style of its training data instead of the correct answer (Shakespeare). This represents a minor but observable form of distributional contamination.

2. **Improved factual accuracy.** The fine-tuned model correctly answered two questions the base model got wrong: identifying the Pacific Ocean as the world's largest (base model said Indian Ocean) and correctly answering "Is water wet?" (base model said no). This may indicate that the reasoning training slightly improved the model's calibration on factual questions, though the small sample size precludes strong conclusions.
