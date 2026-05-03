# Scaled Dot-Product Self-Attention Implementation

![Attention Visualization](images/attention-roll-token.jpg)

*Self-attention weights for the token “Roll" that show how it focuses on contextually relevant words*


## Overview

This project demonstrates how the **self-attention mechanism** works in transformers, which is implemented step-by-step using NumPy.

Our task is to predict the next word in a sentence. This notebook uses the following sentence:

> *“After the Alabama Crimson Tide football team scored a touchdown, the crowd started cheering, ‘Roll ______’”*

Most readers would guess the missing word is **“Tide.”** This is based on the understanding of the context of the sentence.

*(For context: “Roll Tide” is a well-known chant associated with the University of Alabama.)*

To make this prediction, we naturally focus on important words like **“Alabama,” “Crimson,” and “Tide,”** and connect them to a familiar phrase.

A model uses a similar pattern: It figures out which words matter most, understands how they relate, and uses that information to predict the next word. This is the main idea behind how the self-attention mechanism works in modern language models.

This project walks through:
- Positional encodings
- Query, Key, Value (Q, K, V) construction
- Scaled dot-product attention
- Causal masking
- Softmax normalization + temperature scaling
- Attention visualization for individual tokens



## How It Works

### 1. Input Representation

We start with a sequence of token embeddings:

```math
E \in \mathbb{R}^{n \times d}
```
Here, **E** is the matrix of token embeddings, **n** is the number of tokens in the sequence, and **d** is the embedding dimension (the size of each vector).


### 2. Positional Encoding

Attention does not capture word order, so we add positional information to the embeddings.

Without this, **“Roll Tide”** and **“Tide Roll”** would look the same to the model.

We use sinusoidal positional encodings:

```math
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

```math
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

We then combine position and meaning:

```math
X = E + PE
```

**X** is the input used for attention.

### 3. Compute Q, K, V

```math
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
```

Each token is projected into:

- **Query:** what it is looking for
- **Key:** what it offers
- **Value:** what it passes along



### 4. Similarity Scores

Next, we measure how much each word relates to every other word.

We do this by comparing queries and keys:

```math
S = \frac{QK^T}{\sqrt{d_k}}
```

Each value in **S** tells us how relevant one word is to another. Higher values mean stronger relationships.

Here, $d_k$ is the number of dimensions in the key vectors (the size of each key vector).

We divide by $\sqrt{d_k}$ to keep the values from getting too large. Without this scaling, the dot products can grow too large as the number of dimensions increases.


### 5. Causal Masking

When predicting the next word, the model should not be able to see future words.

Therefore, we mask out all future positions in the similarity matrix by setting them to negative infinity:

```python
if j > i:
    S[i, j] = -np.inf
```



### 6. Softmax to Attention Weights

```math
A = \text{softmax}(S)
```

Each row becomes a probability distribution.



### 7. Final Output

```math
\text{Output} = AV
```

Each token becomes a weighted combination of other tokens.


## Tech Stack

- Python
- NumPy
- Matplotlib
- Jupyter Notebook



## How to Run

```bash
git clone https://github.com/stephaniebittner802/scaled-dot-product-self-attention-implementation.git
cd scaled-dot-product-self-attention-implementation
pip install numpy matplotlib notebook
jupyter notebook
```



## Additional Notes
This notebook was designed to provide an educational understanding of how the attention mechanism works. Therefore, the following simplifications were made:

- Although this example does not explicitly predict the next word, the attention pattern reveals which words are most relevant.
- In practice, a trained model would use this information to make predictions, and would likely predict **“Tide”** based on learned patterns.
- The matrices and vectors in this notebook are manually designed for simplification; in real models, these values are learned automatically from data.
