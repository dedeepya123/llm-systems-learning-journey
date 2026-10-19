# Softmax

## What Is Softmax?

Softmax is an activation function that converts raw scores into a probability distribution.

Input:

```text
Logits
```

Output:

```text
Probabilities
```

---

## Why Do We Need Softmax?

Neural networks often produce outputs such as:

```text
[5.2, -1.3, 2.1]
```

These values are called:

```text
Logits
```

Logits are simply scores.

They are not probabilities.

Softmax converts these scores into probabilities that can be interpreted more easily.

---

## Mathematical Definition

For every element:

```text
Softmax(xᵢ)

=

e^(xᵢ)

/

Σ e^(xⱼ)
```

where:

```text
j ranges over all classes
```

---

## Mental Model

Think:

```text
Raw Scores

↓

Softmax

↓

Probability Distribution
```

---

## Example

Input:

```text
[2.0, 1.0, 0.1]
```

Output:

```text
[0.659, 0.242, 0.099]
```

Notice:

```text
All Values ∈ [0,1]

and

Sum = 1
```

---

## Why Exponentiation?

Consider:

```text
[1,2,3]
```

Exponentiation produces:

```text
[e¹,e²,e³]

↓

[2.7,7.4,20.1]
```

This amplifies the differences between logits.

Larger logits receive larger probabilities.

---

## Properties

Softmax guarantees:

```text
Each Output Value

↓

Between 0 and 1
```

and

```text
All Probabilities Sum To 1
```

---

## Common Use Case

Softmax is commonly used for:

```text
Multi-Class Classification
```

Example:

```text
Cat

Dog

Bird
```

The output represents the model's confidence for each class.

---

## Softmax and Cross Entropy

Softmax is usually combined with:

```text
Cross Entropy Loss
```

Pipeline:

```text
Logits

↓

Softmax

↓

Probabilities

↓

Cross Entropy
```

---

## PyTorch Note

In PyTorch:

```python
nn.CrossEntropyLoss()
```

already applies Softmax internally.

Therefore:

```python
logits = model(x)

loss = nn.CrossEntropyLoss()(
    logits,
    target
)
```

Do not manually apply Softmax before CrossEntropyLoss.

---

## Connection To Transformers

Language models output:

```text
Vocabulary Logits
```

Example:

```text
50,000 Token Scores
```

Softmax converts them into:

```text
Token Probabilities
```

These probabilities are used for:

```text
Next Token Prediction
```
