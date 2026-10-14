# Linear Layer Implementation

## Goal

Implement a Linear Layer from scratch and compare it with PyTorch's implementation.

---

## PyTorch API

```python
nn.Linear(
    in_features,
    out_features
)
```

---

## Internal Mental Model

Conceptually:

```python
class Linear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features
    ):

        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features
            )
        )

        self.bias = nn.Parameter(
            torch.empty(
                out_features
            )
        )

    def forward(self, x):

        return (
            x @ self.weight.T
            + self.bias
        )
```

---

## What Happens During Forward?

Input:

```text
(B, in_features)
```

Weight:

```text
(out_features, in_features)
```

Transpose Weight:

```text
(in_features, out_features)
```

Matrix Multiplication:

```text
(B, in_features)

@

(in_features, out_features)

↓

(B, out_features)
```

Bias is then broadcast and added.

Final output:

```text
(B, out_features)
```

---

## Key Takeaway

The Linear Layer is fundamentally:

```text
Matrix Multiplication

+

Bias Addition
```

Everything else is implementation details.
