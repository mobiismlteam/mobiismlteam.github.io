---
layout: article
mathjax: true
toc : true
tags : FewShotLearning
---

# prototypical networks for few-shot learning

## few-shot classification
few-shot classification is a task to train a classifier which can classify example among classes not seen in training with only few labeled examples of each class.

### N-way K-shot classification
- input: $S^1, \cdots, S^N, Q$
    - $S^n = \{ \mathbb{s}^n_1, \cdots, \mathbb{s}^n_K \}, n \in \{1, \cdots, N\}$: class $n$ support set (examples with class $n$)
    - $Q = \{ \mathbb{q}_1, \cdots, \mathbb{q}_M \}$: query set (examples to predict class)
- output: $P$
    - $P = \{ \mathbb{p}_1, \cdots, \mathbb{p}_M \}$: predicted class probabilities

## model

### structure
- $f_\theta(\cdot)$: embedding function
- $D(\cdot, \cdot)$: distance function

prediction
1. **for** $n$ **in** $\{ 1, \cdots, N \}$ **do**
    1. $\mathbb{c}^n \leftarrow \frac{1}{N} \sum\limits_{\mathbb{s} \in S^n} f_\theta(\mathbb{s})$
1. **for** $m$ **in** $\{ 1, \cdots, M \}$ **do**
    1. $\mathbb{e}_m \leftarrow f_\theta(\mathbb{q}_m)$
    1. $\mathbb{d}_m \leftarrow \left( D(\mathbb{e}_m, \mathbb{c}^1), \cdots, D(\mathbb{e}_m, \mathbb{c}^N) \right)$
    1. $\mathbb{p}_m \leftarrow \text{softmin}(\mathbb{d}_m)$

![](protonet.png)

### train
- $N_C$: # way
- $N_S$: # shot
- $N_Q$: # query per way
- loss function: negative log-probability of true class
- train set: $T^1, \cdots, T^U$
    - $T^u = \{ \mathbb{x}^u_1, \cdots, \mathbb{x}^u_V \}, u \in \{ 1, \cdots, U \}$: $V$ examples with class $u$

train is done by iterating over episodes.

train episode
1. $u_1, \cdots, u_{N_C} \leftarrow \text{RandomSample}(\{ 1, \cdots, U \}, N_C)$
1. **for** $i$ **in** $\{ 1, \cdots, N_C \}$ **do**
    1. $S^{u_i} = (\mathbb{s}^{u_i}_1, \cdots, \mathbb{s}^{u_i}_{N_S}) \leftarrow \text{RandomSample}(T^{u_i}, N_S)$
    1. $Q^{u_i} = (\mathbb{q}^{u_i}_1, \cdots, \mathbb{q}^{u_i}_{N_Q}) \leftarrow \text{RandomSample}(T^{u_i} \backslash S^{u_i}, N_Q)$
    1. $\mathbb{c}^{u_i} \leftarrow \frac{1}{N_C} \sum\limits_{\mathbb{s} \in S^{u_i}} f_\theta(\mathbb{s})$
1. **for** $i$ **in** $\{ 1, \cdots, N_C \}$ **do**
    1. **for** $j$ **in** $\{ 1, \cdots, N_Q \}$ **do**
        1. $\mathbb{e}^{u_i}_j \leftarrow f_\theta(\mathbb{q}^{u_i}_j)$
        1. $\mathbb{d}^{u_i}_j \leftarrow \left( D(\mathbb{e}^{u_i}_j, \mathbb{c}^{u_1}), \cdots, D(\mathbb{e}^{u_i}_j, \mathbb{c}^{u_{N_C}}) \right)$
        1. $\mathbb{p}^{u_i}_j \leftarrow \text{softmin}(\mathbb{d}^{u_i}_j)$
        1. $\mathcal{J}^{u_i}_j \leftarrow \ln \mathbb{p}^{u_i}_{j, i}$
1. $\mathcal{J} \leftarrow \frac{1}{N_C N_Q} \sum\limits^{N_C}_{i = 1} \sum\limits^{N_Q}_{j = 1} \mathcal{J}^{u_i}_j$
1. update $\phi$ based on cost $\mathcal{J}$