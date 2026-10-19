# Activation Functions

## What Are Activation Functions?

A Linear Layer transforms input features into a new feature representation.

An Activation Function introduces non-linearity into the network.

Conceptually:

```text
Input

↓

Linear Transformation

↓

Activation Function

↓

Non-Linear Representation
```

---

## Why Do We Need Activations?

Suppose we build:

```text
Linear

↓

Linear

↓

Linear
```

Even though there are multiple layers, this is still equivalent to:

```text
One Larger Linear Transformation
```

This means the network cannot learn complex non-linear relationships.

Activation functions solve this problem by introducing non-linearity.

---

## Mental Model

Think:

```text
Linear Layer

↓

Transforms Features

↓

Activation Function

↓

Decides How Information Flows
```

Together, they become the building block of modern neural networks.

---

## Common Activation Functions

### Sigmoid

```text
f(x) = 1 / (1 + e^(-x))
```

Range:

```text
[0, 1]
```

Useful for:

```text
Binary Classification

Probability Outputs
```

---

### ReLU

```text
f(x) = max(0, x)
```

Range:

```text
[0, ∞)
```

Most commonly used activation in neural networks.

---

### GELU

Used in:

```text
BERT

GPT

Modern Transformers
```

Unlike ReLU, GELU provides smooth gating of information.

---

## Why ReLU Became Popular

ReLU is:

```text
Simple

Computationally Efficient

Works Well With Deep Networks
```

Because of these properties, ReLU became the default activation for many architectures.

---

## Why Activations Matter

Without activations:

```text
Linear

↓

Linear

↓

Linear
```

remains linear.

With activations:

```text
Linear

↓

Activation

↓

Linear
```

the network can model significantly more complex functions.

---

## Connection To Transformers

Transformer Feed Forward Networks contain:

```text
Linear

↓

GELU

↓

Linear
```

Activations remain a critical part of modern LLM architectures.
