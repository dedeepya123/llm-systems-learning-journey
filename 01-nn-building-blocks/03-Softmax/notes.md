# My Notes - Softmax

## Initial Understanding

My first understanding was:

```text
Softmax converts continuous values
into probabilities.
```

While correct, the more precise understanding is:

```text
Softmax converts logits
into a probability distribution.
```

---

## What Clicked For Me

The biggest realization was:

```text
A Neural Network Does Not
Directly Output Probabilities.
```

It outputs:

```text
Scores

or

Logits
```

Example:

```text
[5.2, -1.3, 2.1]
```

These numbers are difficult to interpret.

Softmax converts them into:

```text
Probabilities
```

which are meaningful.

---

## Why Exponential?

Initially the exponential looked strange.

What clicked was:

```text
Higher Scores

↓

Exponentiation

↓

Much Higher Scores
```

So Softmax amplifies confidence.

Example:

```text
1

↓

2.7
```

```text
3

↓

20.1
```

Large logits become much more dominant.

---

## Shape Rule

Softmax does not change shape.

```text
(B, Classes)

↓

(B, Classes)
```

Only the values change.

---

## Important PyTorch Insight

Do not manually apply:

```python
softmax()
```

before:

```python
nn.CrossEntropyLoss()
```

because CrossEntropyLoss already performs this internally.

---

## Connection To Classification

For:

```text
Cat

Dog

Bird
```

The model outputs:

```text
Logits
```

Softmax converts them into:

```text
Class Probabilities
```

---

## Connection To Transformers

A transformer eventually produces:

```text
Vocabulary Logits
```

Example:

```text
50,000 Token Scores
```

Softmax converts them into:

```text
Vocabulary Probabilities
```

Generation algorithms use these probabilities to select the next token.

---

## Explain In One Sentence

Softmax converts model logits into a probability distribution where every probability lies between 0 and 1 and all probabilities sum to 1.
