# Cross Entropy Loss

## What Is Cross Entropy?

Cross Entropy is one of the most commonly used loss functions for classification problems.

Its job is to measure:

```text
How Good

or

How Bad
```

the model's predictions are.

The output of Cross Entropy becomes the error signal that drives learning.

---

## Why Do We Need A Loss Function?

A model can make predictions.

Example:

```text
Cat : 0.80

Dog : 0.15

Bird: 0.05
```

However, predicting is not enough.

The model needs a way to determine:

```text
Was This Prediction Good?

Was It Bad?
```

Cross Entropy answers that question.

---

## Mental Model

Think:

```text
Model Prediction

↓

Cross Entropy

↓

Loss

↓

Backward Pass

↓

Gradients

↓

Learning
```

Without a loss function:

```text
No Error Signal

↓

No Learning
```

---

## Inputs To Cross Entropy

### Predictions

Cross Entropy expects:

```text
Raw Logits
```

Shape:

```text
(B,C)
```

where:

```text
B = Batch Size

C = Number Of Classes
```

Example:

```python
[
    [2.5, 1.0, 0.2]
]
```

---

### Targets

Targets are class indices.

Shape:

```text
(B,)
```

Example:

```python
target = torch.tensor([0])
```

Meaning:

```text
Class 0 is correct
```

---

## Important PyTorch Insight

PyTorch expects:

```text
Raw Logits
```

not:

```text
Softmax Probabilities
```

for:

```python
nn.CrossEntropyLoss()
```

---

## Why?

Because PyTorch already applies:

```text
LogSoftmax
```

internally.

Correct:

```python
logits = model(x)

loss = criterion(
    logits,
    target
)
```

Incorrect:

```python
probs = softmax(
    model(x)
)

loss = criterion(
    probs,
    target
)
```

This applies Softmax twice.

---

## What Does Cross Entropy Reward?

Cross Entropy rewards:

```text
High Probability

For Correct Class
```

and penalizes:

```text
Low Probability

For Correct Class
```

---

Example:

Correct class:

```text
Cat
```

Prediction:

```text
Cat : 0.95
Dog : 0.03
Bird: 0.02
```

Small loss.

Good prediction.

---

Prediction:

```text
Cat : 0.05
Dog : 0.90
Bird: 0.05
```

Large loss.

Bad prediction.

---

## Core Formula

For a single sample:

```text
Loss

=

-log(
    p_correct
)
```

Where:

```text
p_correct
```

is the probability assigned to the correct class.

---

## Example

Correct class probability:

```text
0.99
```

Loss:

```text
-log(0.99)

≈ 0.01
```

Very small loss.

---

Correct class probability:

```text
0.10
```

Loss:

```text
-log(0.10)

≈ 2.30
```

Large loss.

---

Correct class probability:

```text
0.001
```

Loss:

```text
≈ 6.90
```

Very large loss.

---

## Connection To Softmax

Conceptually:

```text
Logits

↓

Softmax

↓

Probabilities

↓

Cross Entropy

↓

Loss
```

This is the foundation of most classification systems.

---

## Shape Reasoning

Predictions:

```text
(B,C)
```

Example:

```text
(32,10)
```

---

Targets:

```text
(B,)
```

Example:

```text
(32,)
```

containing:

```text
0
4
2
9
...
```

class indices.

---

## Common Use Cases

Cross Entropy is used in:

```text
Image Classification

Object Classification

Language Models

Token Prediction

Sequence Classification
```

---

## Connection To Transformers

Language models produce:

```text
Vocabulary Logits
```

Example:

```text
50,000 Token Scores
```

Cross Entropy measures:

```text
How Much Probability
Was Assigned To
The Correct Next Token
```

Training GPT-style models ultimately depends on Cross Entropy Loss.

## What Cross Entropy Actually Uses

A common misconception is:

```text
Take Highest Probability Class

↓

Compute Loss
```

This is NOT how Cross Entropy works.

Cross Entropy does not use:

```python
argmax()
```

during training.

Instead it asks:

```text
What probability
did the model assign
to the correct class?
```

Example:

Prediction:

```text
[0.40, 0.30, 0.20, 0.05, 0.05]
```

Target:

```text
Class 4
```

Cross Entropy selects:

```text
0.05
```

because class 4 is correct.

Loss:

```text
-log(0.05)
```

The lower the probability assigned to the correct class, the larger the loss.

The higher the probability assigned to the correct class, the smaller the loss.
