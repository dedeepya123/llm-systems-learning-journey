# Softmax Implementation

## Goal

Implement Softmax from scratch and compare it against PyTorch.

---

## Mathematical Definition

```text
Softmax(xᵢ)

=

e^(xᵢ)

/

Σ e^(xⱼ)
```

---

## Naive Implementation

```python
class SoftmaxScratch(nn.Module):

    def forward(self, x):

        exp_x = torch.exp(x)

        return (
            exp_x
            /
            exp_x.sum(
                dim=-1,
                keepdim=True
            )
        )
```

---

## Numerical Stability Problem

Consider:

```text
x = 1000
```

Then:

```python
torch.exp(1000)
```

may overflow.

---

## Stable Implementation

A common trick:

```text
Subtract Maximum Logit
```

before exponentiation.

---

```python
class SoftmaxScratch(nn.Module):

    def forward(self, x):

        x = (
            x
            -
            x.max(
                dim=-1,
                keepdim=True
            )[0]
        )

        exp_x = torch.exp(x)

        return (
            exp_x
            /
            exp_x.sum(
                dim=-1,
                keepdim=True
            )
        )
```

---

## Why This Works

Softmax is shift invariant.

Example:

```text
[10,11,12]
```

and

```text
[-2,-1,0]
```

produce the same probabilities.

---

## Shape Behavior

Input:

```text
(B, Num Classes)
```

Output:

```text
(B, Num Classes)
```

Softmax changes values but not shapes.

---

## Example

Input:

```text
(32,10)
```

Output:

```text
(32,10)
```
