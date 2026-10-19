# Verification Against PyTorch

## Goal

Verify that custom implementations behave the same as PyTorch reference implementations.

---

# ReLU Verification

Compare:

```python
ReLUScratch()
```

against:

```python
nn.ReLU()
```

Input:

```python
x = torch.randn(
    32,
    128
)
```

Verify:

```python
torch.allclose(
    scratch_output,
    torch_output
)
```

Expected:

```text
True
```

---

# Sigmoid Verification

Compare:

```python
SigmoidScratch()
```

against:

```python
nn.Sigmoid()
```

Verify:

```python
torch.allclose(
    scratch_output,
    torch_output
)
```

Expected:

```text
True
```

---

# GELU Verification

Compare:

```python
GELUScratch()
```

against:

```python
nn.GELU()
```

Allow small floating-point differences.

Verify:

```python
torch.allclose(
    scratch_output,
    torch_output,
    atol=1e-5
)
```

Expected:

```text
True
```

---

# Shape Verification

Input:

```text
(32,128)
```

Expected Output:

```text
(32,128)
```

for all activation functions.

---

# Success Criteria

```text
Output Shapes Match

Output Values Match

Gradient Flow Works
```

If all checks pass, implementation is considered correct.
