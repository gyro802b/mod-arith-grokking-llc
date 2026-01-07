# Modular Arithmetic Grokking: LLC Estimation Across Architectures

A technical exploration applying Local Learning Coefficient (LLC) estimation to the grokking phenomenon in modular arithmetic, comparing behavior between a Constant Attention and Transformer model.

## What This Is

This is a **learning project** exploring the intersection of Singular Learning Theory and neural network generalization. I implemented LLC estimation using the [devinterp](https://github.com/timaeus-research/devinterp) library and replicated the architectural setup from [The Clock and the Pizza](https://arxiv.org/abs/2306.17844) paper.

**Scope**: Single-seed exploration with one run per architecture. This is not a rigorous study, the goal was to build implementation familiarity and intuition for these techniques. As a learning project this was timeboxed and compute-boxed to my T4 Colab credits.

## Background

**Grokking** is a phenomenon where neural networks suddenly generalize long after memorizing training data. [Power et al. (2022)](https://arxiv.org/abs/2201.02177) first observed this on modular arithmetic tasks.

**The Local Learning Coefficient (LLC)** from Singular Learning Theory provides a geometry-aware measure of model complexity. Unlike parameter counts, LLC captures the effective dimensionality of the loss landscape near a solution. See [Lau et al. (2023)](https://arxiv.org/abs/2308.12108) for the formal treatment.

**Clock vs Pizza**: [Zhong et al. (2023)](https://arxiv.org/abs/2306.17844) showed that different architectures learn different algorithms for modular addition. Transformers tend to learn "Clock" (Fourier-based) while MLPs learn "Pizza" (slice-based) algorithms, distinguishable via gradient symmetry and distance irrelevance metrics.

## What I Implemented

- **Three model architectures** for modular addition `(a + b) % p`:
  - `Transformer`: Single-layer with learned positional encodings (expected: Clock algorithm)
  - `ConstAttn Transformer`: Constant attention variant (expected: Pizza algorithm)
  - `MLP`: Single hidden layer baseline

- **LLC estimation** via SGLD sampling at training checkpoints

- **Algorithm detection metrics** from the Pizza/Clock paper:
  - Gradient Symmetry
  - Distance Irrelevance

- **Training infrastructure**: Base models, LLC estimation, ablations, visualization

## Repository Structure

```
├── Modular_Arithmetic_Grokking_LLC_inspection.ipynb  # Main experiment notebook
├── Modular_Arithmetic_Grokking_Ablations.ipynb       # Ablation experiment notebook
├── plots/                                            # Generated visualizations
│   ├── model_a_llc_vs_loss.png
│   ├── model_b_llc_vs_loss.png
│   └── ...
└── README.md
```

## Observations

*Example training run (Transformer):*

![LLC vs Loss for Transformer](plots/model_b_llc_vs_loss.png)

All three architectures grok early (within the first few thousand steps). LLC doesn't settle to a stable value post-grokking. It continues to drift, so I tracked minimums alongside final results. I wouldn't read too much into the specific values given the single-seed setup.

The algorithm metrics (gradient symmetry, distance irrelevance) do differentiate the architectures as expected from the Pizza/Clock paper.

## Running the Notebook

```bash
# Requires GPU runtime (Colab T4 works)
pip install devinterp torch matplotlib pandas tqdm
```

Open in [Google Colab](https://colab.research.google.com/github/gyro802b/mod-arith-grokking-llc/blob/main/Modular_Arithmetic_Grokking_LLC_inspection.ipynb) and run all cells. Full training + LLC estimation takes a few hours on a T4 (~45 minutes per model)

## Experiment Configuration

```python
# Training
p = 59                    # Modulus
train_frac = 0.6          # Train/test split
n_batches = 25000         # Full-batch training steps
lr = 0.001                # Learning rate (0.01 for MLP)
weight_decay = 2.0        # AdamW regularization

# LLC Estimation (per checkpoint)
num_draws = 500           # SGLD samples
num_burnin_steps = 2000   # Burn-in before sampling
gamma = 100               # Localization strength
```

Parameters follow the Pizza/Clock paper where applicable.

## Limitations

- **Single seed**: Results may not be representative. Proper validation would require 3-5+ seeds with confidence intervals.
- **No determinism guarantees**: DataLoader shuffling and CUDA operations introduce non-determinism across environments. This is less of a problem when performing full-batch training, but for batch size ablations this is a known issue.
- **LLC estimation variance**: SGLD-based estimation has inherent noise; more chains/draws would improve reliability.

## What Could Be Done With More Compute & Time

1. Run 5 seeds per architecture with error bars on all metrics
2. Sweep LLC estimation hyperparameters (gamma, learning rate, num_draws)
3. Add more algorithm metrics from Clock/Pizza paper (attention patterns, PCAs)
4. Track LLC evolution at finer granularity around the grokking transition

## References

- [Power et al. (2022)](https://arxiv.org/abs/2201.02177) - Grokking: Generalization Beyond Overfitting
- [Zhong et al. (2023)](https://arxiv.org/abs/2306.17844) - The Clock and the Pizza
- [Lau et al. (2023)](https://arxiv.org/abs/2308.12108) - Quantifying Degeneracy in Singular Models via the Learning Coefficient
- [devinterp library](https://github.com/timaeus-research/devinterp) - LLC estimation toolkit
