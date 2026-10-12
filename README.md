# Text Embeddings & Semantic Similarity

A beginner-friendly hands-on guide and implementation of **Text Embeddings** and **Cosine Similarity** using Python, NumPy, and `sentence-transformers`.

---

## What is a Text Embedding?

Computers cannot inherently understand the meaning of human language; they only see strings of characters. **Text embeddings** solve this by transforming words, sentences, or documents into dense numerical vectors (arrays of real numbers).

```text
"Hello, how are you?"  ──[ Embedding Model ]──>  [-0.042, 0.081, 0.012, ..., 0.035]
                                                   (Dense 384-dimensional vector)
```

### The Key Idea: Semantic Space
Embedding models place sentences in a high-dimensional geometric space where **distance and angle represent meaning**:
* Sentences with **similar meanings** are mapped close together.
* Sentences with **different meanings** are mapped far apart.
* They understand context and synonyms—not just exact keyword matches!

---

## What is Cosine Similarity?

To measure how similar two text embeddings are, we calculate the **Cosine Similarity** between their vectors.

Cosine similarity evaluates the cosine of the angle $\theta$ between two vectors in multi-dimensional space:

$$\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

### How to Interpret the Score:
| Score Range | Interpretation | Example |
| :--- | :--- | :--- |
| **0.80 - 1.00** | **Very High Similarity** (near identical intent or paraphrasing) | *"How are you?"* vs *"How are you doing today?"* |
| **0.50 - 0.79** | **Moderate Similarity** (related topics or domain) | *"I love programming in Python"* vs *"Software engineering is fun"* |
| **0.20 - 0.49** | **Low Similarity** (slight overlap or common stopwords) | *"The sun is shining"* vs *"I ate an apple for lunch"* |
| **< 0.20** | **Unrelated / Orthogonal** | *"Hello, how are you?"* vs *"Cats are playful animals"* |

---

## Model Used: `all-MiniLM-L6-v2`

This project uses the [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model:
* **Vector Dimension**: `384` floating-point numbers per sentence.
* **Speed & Performance**: Extremely fast and lightweight, making it one of the most popular choices for sentence embeddings, clustering, and semantic search.

---

## Project Structure

```text
Embedding/
├── pyproject.toml              # Project dependencies and packaging configuration
├── uv.lock                     # Locked dependency versions
├── README.md                   # Project documentation & explanation
└── src/
    └── embedding/
        ├── __init__.py         # Package entry point
        └── embed.py            # Main embedding & cosine similarity script
```

---

## Getting Started

### Prerequisites
* Python `>= 3.10`
* [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### 1. Installation

Using `uv`:
```bash
# Clone the repository
git clone https://github.com/Prajwal-0x10/Embedding.git
cd Embedding

# Install dependencies into virtual environment
uv sync
```

Alternatively using `pip`:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install numpy sentence-transformers
```

---

## Running the Script

Execute the embedding script:

```bash
uv run python src/embedding/embed.py
```
*(or run `python src/embedding/embed.py` inside your active virtual environment)*

### Sample Output

```text
Loading embedding model 'all-MiniLM-L6-v2'...
Model loaded successfully!

============================================================
DEMO 1: Inspecting an Embedding Vector
============================================================
Input text   : "This is a sample sentence to examine its embedding vector."
Vector shape : (384,)  -> 384 numbers representing this text
Vector type  : <class 'numpy.ndarray'>
First 5 dims : [-0.0341  0.0512 -0.0128  0.0743 -0.0215] ...

============================================================
DEMO 2: Semantically Similar Sentences
============================================================
Sentence 1 : "Hello, how are you?"
Sentence 2 : "How are you doing today?"
Score      : 0.8643
Meaning    : Very High Similarity (almost identical meaning)

============================================================
DEMO 3: Semantically Unrelated Sentences
============================================================
Sentence 1 : "Hello, how are you?"
Sentence 2 : "Cats are very playful animals."
Score      : 0.0825
Meaning    : Very Low / No Similarity (unrelated)

============================================================
DEMO 4: Paraphrased Sentences (Different words, same meaning)
============================================================
Sentence 1 : "The weather outside is wonderful and sunny."
Sentence 2 : "It is a beautiful and bright day outdoors."
Score      : 0.8126
Meaning    : Very High Similarity (almost identical meaning)
```

---

## Where Are Embeddings Used in AI?

1. **Retrieval-Augmented Generation (RAG)**: Finding the most relevant documents in a vector database before sending them to an LLM.
2. **Semantic Search**: Searching by concept or intent rather than exact keyword matches.
3. **Recommendation Systems**: Matching user queries or profiles with relevant content.
4. **Clustering & Classification**: Grouping customer reviews, support tickets, or news articles by topic.
