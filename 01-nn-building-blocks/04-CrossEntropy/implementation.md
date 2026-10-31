# Cross Entropy Implementation

## Goal

Implement Cross Entropy Loss from scratch.

---

## Building Blocks

Cross Entropy can be viewed as:

```text
Softmax

+

Negative Log Likelihood
```

---

## Step 1

Convert logits into probabilities.

```python
probs = softmax(
    logits
)
```

---

## Step 2

Select probability of correct class.

Example:

```text
Probabilities

[0.80, 0.15, 0.05]

Target Class

0
```

Selected probability:

```text
0.80
```

---

## Step 3

Apply negative log.

```python
loss = -torch.log(
    p_correct
)
```

---

## Scratch Implementation

```python
class CrossEntropyScratch(
    nn.Module
):

    def forward(
        self,
        logits,
        target
    ):

        probs = torch.softmax(
            logits,
            dim=-1
        )

        p_correct = probs[
            torch.arange(
                target.shape[0]
            ),
            target
        ]

        loss = -torch.log(
            p_correct
        )

        return loss.mean()
```

---

## Important Note

PyTorch uses a significantly more numerically stable implementation internally.

This implementation is intended only for learning purposes.
