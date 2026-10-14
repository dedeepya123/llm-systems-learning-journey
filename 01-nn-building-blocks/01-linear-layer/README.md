# Linear Layer

## What is a Linear Layer?

A Linear Layer is one of the fundamental building blocks of a Neural Network.

Its job is to transform input features into a new output feature space using learned parameters.

Mathematically:

```text
Y = WX + b
```

Where:

```text
X → Input Features

W → Learned Weights

b → Learned Bias

Y → Output Features
```

As training progresses, the values of W and b are updated so that the transformation becomes useful for the task being solved.

---

## Mental Model

I think of a Linear Layer as:

```text
Input Features

↓

Weighted Mixing

↓

Output Features
```

Every output feature is computed as a weighted combination of all input features.

---

## Why Do We Need It?

Neural networks need a mechanism to transform information from one representation into another.

Example:

```text
Raw Features

↓

Linear Layer

↓

Learned Representation
```

The output features are not necessarily human interpretable.

They become internal representations that later layers can process.

---

## Shape Reasoning

Input:

```text
(B, in_features)
```

Weight:

```text
(out_features, in_features)
```

Bias:

```text
(out_features)
```

Output:

```text
(B, out_features)
```

Example:

```python
nn.Linear(4, 3)
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

## Why Is Bias Needed?

Without bias:

```text
Output = WX
```

With bias:

```text
Output = WX + b
```

Bias allows shifting the output independently of the input values.

This gives the model additional flexibility.

---

## Why Is It Called Linear?

The computation consists of:

```text
Matrix Multiplication

+

Bias Addition
```

There is no activation function inside the Linear Layer.

This means:

```text
Linear

↓

Linear

↓

Linear
```

is still a linear transformation.

This is why activation functions such as ReLU are required.

---

## Connection To Transformers

Linear layers appear everywhere inside transformers.

Examples:

```text
Q Projection

K Projection

V Projection

Output Projection

Feed Forward Layers

LM Head
```

Understanding Linear Layers is the first step toward understanding transformers.
