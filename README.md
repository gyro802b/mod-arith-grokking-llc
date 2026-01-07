# Modular Arithmetic Pizza/Clock Exploration

An exploratory project for estimating the Local Learning Coefficient (LLC) across various toy models of modular arithmetic, which learn different algorithms.

First ever DevInterp project! Exploratory project investigating LLC dynamics during grokking of different model types

Pretty rough, but was timeboxed as a learning exercise; future work should extend ablations, test on more seeds, collect more data, and improve SGLD hyperparameter tuning.

### Notebooks

- [Main LLC inspection](Modular_Arithmetic_Grokking_LLC_inspection.ipynb)
- [Additional Ablations](Modular_Arithmetic_Grokking_Ablations.ipynb)

### Background

- [Grokking paper](https://www.alignmentforum.org/posts/4v3hMuKfsGatLXPgt/investigating-the-learning-coefficient-of-modular-addition)
- [LLC / devinterp](https://arxiv.org/abs/2308.12108)
- [Pizza and Clock paper](https://arxiv.org/abs/2306.17844)

In this notebook we investigate how the local learning coefficient (LLC) evolves after grokking modular arithmetic for different architectures (Transformer, MLP, Const Attention).

The authors of https://arxiv.org/pdf/2306.17844 have found that models can learn different algorithms to perform modular arithmetic, and it is an open question on how the LLC is impacted by them.

### Scope
3 models are trained to perform modular arithmetic: The models are specifically trained to learn the operation `(a+b) % p`, where `a,b` are inputs, and `p` is fixed (in this notebook, `p=59`)

1. A single layer transformer with learned positional encodings and constant attention. Corresponds to `ModelA` in Pizza/Clock.
2. A single layer transformer with learned positional encodings. Corresponds to `ModelB` in Pizza/Clock.
3. A single layer MLP.

Main notebook performs a single training run of each with a fixed seed for reproducability. Ablations use separate seeds and hyperparamters.

### Observations

#### LLC Estimates (lambdahat) and Loss curves for ModelA (Constant Attention) and ModelB (Regular Transformer):

![LLC Estimates And Loss For Const Attn](plots/model_a_llc_vs_loss.png)

![LLC Estimates And Loss For Regular Transformer](plots/model_b_llc_vs_loss.png)

ModelA and ModelB learn different algorithms, and in both, LLC drops sharply during initial grokking, and more slowly over extended training until a sharp rise after a very long period of time.

#### Notes

- In all 3 model types, the LLC correlates with loss.
- In the Transformer and Const Attn models, the LLC correlated with changes in algorithm metrics.
- In the Transformer and Const Attn models, the LLC reached minimums of approximately 39 and 49 at checkpoints 73 and 75 respectively for this run; Additional seeds should be tested for more confidence in results.
- When trained long enough, the Transformer and Const Attn models appear to undergo a second phase change, post-grokking, where the LLC rises. Both models final LLCs were similar at approximately 115 and 117; 
- For transformer models the LLC decreased if we lowered parameter count or disabled positional encoding. Between similarly sized constAttn vs regular transformer, LLC variance seems low.
- MLP was likely overtrained, and hyperparameter selection for SGLD could be improved. My LLC estimate on the MLP model is near 0, which is a strange result. (The [reference](https://github.com/timaeus-research/devinterp/blob/main/examples/grokking.ipynb) finds it is closer to 23, under more grounded training conditions).

#### The Local Learning Coefficient (LLC)

The LLC arises from the field of singular learning theory and serves as a general metric of "model complexity" in a neural network.

The [formal definition of the LLC](https://arxiv.org/abs/2308.12108) is quite dense and gets into the realm of algebraic geometry. For our purposes, we can regard the LLC as "lower LLC = less dimensions used by the model".

For most models, the LLC must be estimated, ironically enough, through machine learning techniques such as SGLD.

#### This all seems pretty informal, how could this be turned into something more substantial?

Well, this was a learning project for me to get familiar with the ecosystem. But if I had to continue:

1. Fix code quality: There's a lot missing. No checkpointing currently, everything packed into a colab notebook, ablation code is unpolished, SGLD hyperparameters could be better, ablation could be parallelized, everything could be logged and exported better.
2. Implement full analysis from Pizza/Clock: Currently I'm just placing my faith in the 2 core algorithm metrics. There's a lot more that Pizza/Clock did.
3. Perform *many* more ablations on ConstAttn vs Regular Transformer. If there is some pattern in how LLC varies across model types, it should be durable through ablations, provided that the algorithms learned remain similar.
