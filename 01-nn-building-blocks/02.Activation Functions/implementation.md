# Activation Function Implementations

## Goal

Implement common activation functions from scratch and compare them with PyTorch implementations.

---

# Sigmoid

Mathematically:

```text
σ(x) = 1 / (1 + e^(-x))
```

Implementation:

```python
class SigmoidScratch(nn.Module):

    def forward(self, x):

        return (
            1 /
            (
                1 +
                torch.exp(-x)
            )
        )
```

---

# ReLU

Mathematically:

```text
ReLU(x) = max(0, x)
```

Implementation:

```python
class ReLUScratch(nn.Module):

    def forward(self, x):

        return torch.maximum(
            torch.zeros_like(x),
            x
        )
```

Alternative:

```python
torch.clamp(
    x,
    min=0
)
```

---

# GELU

Modern transformer activation.

Approximation:

```python
class GELUScratch(nn.Module):

    def forward(self, x):

        return (
            0.5
            * x
            * (
                1 +
                torch.tanh(
                    math.sqrt(
                        2 / math.pi
                    )
                    * (
                        x
                        + 0.044715
                        * x**3
                    )
                )
            )
        )
```

---

# Input And Output Shapes

Input:

```text
Any Shape
```

Output:

```text
Same Shape
```

Activations modify values but do not modify tensor shapes.

Example:

```text
(32,128)

↓

Activation

↓

(32,128)
```

---

# Key Takeaway

Activation functions:

```text
Change Values

Not Shapes
```
