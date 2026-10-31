# Verification Against PyTorch

## Goal

Verify that the custom implementation behaves similarly to PyTorch's implementation.

---

## Reference

```python
nn.CrossEntropyLoss()
```

---

## Input

```python
logits = torch.randn(
    32,
    10
)

targets = torch.randint(
    0,
    10,
    (32,)
)
```

---

## Compare Loss Values

```python
scratch_loss = scratch(
    logits,
    targets
)

torch_loss = criterion(
    logits,
    targets
)
```

---

Verify:

```python
torch.allclose(
    scratch_loss,
    torch_loss,
    atol=1e-5
)
```

Expected:

```text
True
```

---

## Shape Verification

Logits:

```text
(32,10)
```

Targets:

```text
(32,)
```

Loss:

```text
Scalar
```

Expected:

```text
One Loss Value
For Entire Batch
```

---

## Success Criteria

```text
Loss Value Matches

Expected Shapes Match

Behavior Matches
```
