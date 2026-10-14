# My Notes - Linear Layer

## Initial Understanding

My first mental model was:

```text
Input Features

↓

Transformation

↓

Output Features
```

The layer learns how to transform one feature representation into another.

---

## What Clicked For Me

The biggest realization was:

```text
nn.Module

=

Computation

+

Parameters
```

For a Linear Layer:

Computation:

```text
Y = WX + b
```

Parameters:

```text
W

b
```

---

## Understanding The Weight Matrix

Initially I thought:

```text
W learns patterns
```

A better way to think about it is:

```text
W stores learned parameters
```

The patterns discovered during training are encoded inside the values of W.

---

## Understanding Bias

Initially I understood the weight matrix but not the bias.

Now I think of bias as:

```text
Output Shifting
```

It allows the layer to produce useful outputs even when inputs are small or zero.

---

## Shape Rule I Want To Remember

```text
Linear(in_features, out_features)

(B, in_features)

↓

(B, out_features)
```

Example:

```python
nn.Linear(4,3)
```

Input:

```text
(32,4)
```

Output:

```text
(32,3)
```

---

## PyTorch Mental Model

When I create:

```python
nn.Linear(...)
```

PyTorch:

```text
Creates Parameters

↓

Registers Parameters

↓

Initializes Parameters

↓

Ready For Training
```

---

## Connection To Backpropagation

The Linear Layer itself does not learn.

Training updates:

```text
Weight

Bias
```

through:

```text
Forward Pass

↓

Loss

↓

Backward Pass

↓

Optimizer Step
```

---

## Connection To Transformers

One thing I want to remember:

Whenever I see:

```python
nn.Linear(...)
```

inside a transformer,

I should not think:

```text
Magic Transformer Logic
```

Instead think:

```text
Feature Transformation
```

Q, K and V projections are Linear Layers.

Feed Forward Networks are Linear Layers.

The LM Head is another Linear Layer.

---

## Explain In One Sentence

A Linear Layer learns weights and biases that transform input features into a new feature representation that is more useful for solving a task.
