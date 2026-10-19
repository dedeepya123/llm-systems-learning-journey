# Verification Against PyTorch

## Goal

Verify that the custom Softmax implementation matches PyTorch's implementation.

---

## Reference Layer

```python
nn.Softmax(
    dim=-1
)
```

---

## Test Input

```python
x = torch.randn(
    32,
    10
)
```

---

## Compare Outputs

```python
scratch_out = scratch(x)

torch_out = torch_softmax(x)
```

---

Verify:

```python
torch.allclose(
    scratch_out,
    torch_out,
    atol=1e-6
)
```

Expected:

```text
True
```

---

## Probability Check

Verify:

```python
scratch_out.sum(
    dim=-1
)
```

Expected:

```text
All Rows Sum To 1
```

---

## Shape Check

Input:

```text
(32,10)
```

Output:

```text
(32,10)
```

Expected:

```text
Shape Preserved
```

---

## Numerical Stability Check

Input:

```python
x = torch.tensor(
    [[1000.0, 1001.0, 1002.0]]
)
```

Stable Softmax should still produce valid probabilities.

---

## Success Criteria

```text
Output Matches PyTorch

Probabilities Sum To 1

Shape Preserved

Stable For Large Logits
```
