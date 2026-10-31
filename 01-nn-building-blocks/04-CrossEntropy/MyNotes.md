# My Notes - Cross Entropy Loss

## Initial Understanding

My first understanding was:

```text
Cross Entropy takes model predictions
and computes a loss value.
```

This is correct but incomplete.

The bigger realization was understanding exactly what prediction information is used to compute the loss.

---

## What Clicked For Me

The model does not learn from predictions.

The model learns from:

```text
Loss
```

Pipeline:

```text
Logits

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

Cross Entropy converts prediction quality into an optimization signal.

---

## Logits vs Probabilities

Neural networks do not directly output probabilities.

They output:

```text
Logits
```

Example:

```text
[2.3, 1.1, 0.4, -0.5, 3.2]
```

These values are simply scores.

To interpret them as probabilities:

```text
Logits

↓

Softmax

↓

Probabilities
```

Example:

```text
[0.23, 0.07, 0.03, 0.01, 0.66]
```

Now every value:

```text
0 ≤ p ≤ 1
```

and:

```text
Sum = 1
```

---

## Understanding Prediction Shape

Suppose:

```text
Batch Size = 10

Number Of Classes = 5
```

Predictions:

```text
(10,5)
```

Interpretation:

```text
10 Samples

5 Class Scores Per Sample
```

Example:

```text
Sample 0

[2.1, 0.5, 1.4, 0.2, 3.0]

Sample 1

[0.4, 2.2, 1.0, 0.8, 0.3]
```

and so on.

---

## Understanding Target Shape

Targets:

```text
(10,)
```

Example:

```python
tensor([
    4,
    1,
    0,
    3,
    2,
    4,
    1,
    0,
    2,
    3
])
```

Meaning:

```text
Sample 0 -> Class 4 is correct

Sample 1 -> Class 1 is correct

Sample 2 -> Class 0 is correct
```

One target value per sample.

---

## Biggest Confusion I Had

Initially I thought:

```text
Take Softmax

↓

Find Highest Probability Class

↓

Compute Loss
```

This is wrong.

That logic is used during:

```text
Inference

Accuracy Calculation
```

but not during Cross Entropy.

---

## What Cross Entropy Actually Uses

Cross Entropy asks:

```text
What probability
did the model assign
to the correct class?
```

Example:

Probabilities:

```text
[0.10, 0.10, 0.10, 0.20, 0.50]
```

Target:

```text
4
```

Correct class probability:

```text
0.50
```

Cross Entropy uses:

```text
0.50
```

for loss computation.

---

Another example:

Probabilities:

```text
[0.40, 0.30, 0.20, 0.05, 0.05]
```

Target:

```text
4
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

Large loss because the model was not confident in the correct answer.

---

## Understanding The Extraction Logic

This line initially confused me:

```python
p_correct = probs[
    torch.arange(
        targets.shape[0]
    ),
    targets
]
```

Suppose:

```python
probs =
[
 [0.1,0.1,0.1,0.2,0.5],

 [0.2,0.5,0.1,0.1,0.1],

 [0.7,0.1,0.1,0.05,0.05]
]

targets =
[
 4,
 1,
 0
]
```

This becomes:

```python
[
 probs[0,4],
 probs[1,1],
 probs[2,0]
]
```

Result:

```python
[
 0.5,
 0.5,
 0.7
]
```

These are the probabilities assigned to the correct class for each sample.

---

## Core Formula

For one sample:

```text
Loss

=

-log(
    p_correct
)
```

Examples:

```text
p_correct = 0.99

↓

Loss ≈ 0.01
```

Good prediction.

---

```text
p_correct = 0.10

↓

Loss ≈ 2.30
```

Bad prediction.

---

```text
p_correct = 0.001

↓

Loss ≈ 6.90
```

Very bad prediction.

---

## Important PyTorch Insight

PyTorch:

```python
nn.CrossEntropyLoss()
```

expects:

```text
Raw Logits
```

not:

```text
Softmax Probabilities
```

because Softmax is already applied internally.

Correct:

```python
loss = criterion(
    logits,
    targets
)
```

Incorrect:

```python
loss = criterion(
    softmax(logits),
    targets
)
```

---

## Shape Rule

Predictions:

```text
(B,C)
```

Targets:

```text
(B,)
```

Loss:

```text
Scalar
```

---

## Connection To Transformers

A language model predicts:

```text
Vocabulary Logits
```

Example:

```text
50,000 Token Scores
```

Cross Entropy asks:

```text
How much probability
was assigned
to the correct next token?
```

GPT training is fundamentally minimizing Cross Entropy Loss.

---

## Biggest Lesson

Cross Entropy does NOT care about the predicted class.

Cross Entropy cares about:

```text
How much probability
was assigned
to the correct class.
```

That single idea made the loss function finally click for me.

---

## Explain In One Sentence

Cross Entropy converts the probability assigned to the correct class into a numerical error signal that can be optimized through backpropagation.
