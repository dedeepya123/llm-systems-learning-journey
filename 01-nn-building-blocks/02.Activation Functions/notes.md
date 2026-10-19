# My Notes - Activation Functions

## Initial Understanding

My first understanding was:

```text
Linear Layers Transform Features

Activation Functions Capture Non-Linearity
```

A better way to think about it:

```text
Linear Layers Transform Information

Activation Functions Introduce Non-Linearity
```

---

## What Clicked For Me

The biggest realization was:

```text
Linear

↓

Linear

↓

Linear
```

can always be reduced to:

```text
One Bigger Linear Layer
```

This means depth alone does not increase expressive power.

The reason neural networks work is because activations are inserted between Linear Layers.

---

## Linear + Activation

I now think about neural networks as:

```text
Linear

+

Activation
```

repeated many times.

This is the fundamental building block of deep learning.

---

## Understanding ReLU

ReLU behaves like:

```text
Negative Values

↓

0

Positive Values

↓

Pass Through
```

Example:

```text
-5

↓

0
```

```text
3

↓

3
```

It acts like a gate.

---

## Understanding Sigmoid

Sigmoid maps values into:

```text
[0,1]
```

which makes it useful for probabilities.

Large negative inputs become:

```text
~0
```

Large positive inputs become:

```text
~1
```

---

## Understanding GELU

For now I think of GELU as:

```text
A Smoother Version Of ReLU
```

Instead of completely zeroing information, it smoothly scales it.

This is why it became popular in transformers.

---

## Shape Intuition

One thing I want to remember:

```text
Activations

Do Not Change Shapes
```

They only change:

```text
Values
```

Example:

```text
(32,128)

↓

ReLU

↓

(32,128)
```

---

## PyTorch Mental Model

Activations usually contain:

```text
Computation
```

but not:

```text
Parameters
```

Example:

```python
nn.ReLU()
```

has:

```text
No Learnable Weights
```

Unlike:

```python
nn.Linear()
```

which contains:

```text
Weights

Biases
```

---

## Connection To Transformers

One important realization:

Transformer Feed Forward Networks are:

```text
Linear

↓

GELU

↓

Linear
```

So even in GPT and BERT, activations remain a core building block.

---

## Explain In One Sentence

Activation functions introduce non-linearity into neural networks, allowing them to learn relationships that cannot be represented by linear transformations alone.
