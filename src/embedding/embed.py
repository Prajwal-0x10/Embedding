"""
=============================================================================
Understanding Text Embeddings & Cosine Similarity
=============================================================================
What is an Embedding?
---------------------
An embedding is a way to convert human language (words, sentences, or paragraphs)
into a dense list of numbers (a vector).

Key intuition:
- Computers cannot directly understand the meaning of words.
- An embedding model maps text into a multi-dimensional "semantic space".
- Sentences with similar meanings are placed close together in this space.
- Sentences with different meanings are placed far apart.

What is Cosine Similarity?
--------------------------
Cosine similarity measures the angle between two multi-dimensional vectors.
- Value = 1.0  -> Pointing in the exact same direction (semantically identical)
- Value = 0.0  -> Orthogonal / 90 degrees (semantically unrelated)
- Value = -1.0 -> Pointing in opposite directions (diametrically opposite)

Formula:
    cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
    Where:
      - (A · B) is the dot product.
      - ||A|| and ||B|| are the Euclidean magnitudes (norms) of each vector.
=============================================================================
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two 1D embedding vectors.

    Parameters:
        emb1 (np.ndarray): Embedding vector for the first text.
        emb2 (np.ndarray): Embedding vector for the second text.

    Returns:
        float: Cosine similarity score between -1.0 and 1.0.
    """
    # Step 1: Calculate the dot product (measures directional alignment)
    dot_product = np.dot(emb1, emb2)

    # Step 2: Calculate Euclidean norms (lengths) of both vectors
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)

    # Step 3: Handle edge case to avoid division by zero
    if norm_emb1 == 0 or norm_emb2 == 0:
        return 0.0

    # Step 4: Divide dot product by the product of vector magnitudes
    similarity = dot_product / (norm_emb1 * norm_emb2)
    return float(similarity)


# Backward-compatibility alias for previous typo
cosine_simalarity = cosine_similarity


def interpret_similarity(score: float) -> str:
    """Provides a human-friendly description for a similarity score."""
    if score >= 0.80:
        return "Very High Similarity (almost identical meaning)"
    elif score >= 0.50:
        return "Moderate Similarity (related topics/context)"
    elif score >= 0.20:
        return "Low Similarity (weak connection)"
    else:
        return "Very Low / No Similarity (unrelated)"


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Load the Pre-trained Embedding Model
    # -------------------------------------------------------------------------
    # 'all-MiniLM-L6-v2' is a lightweight, fast, and high-quality model
    # from Sentence-Transformers.
    # Note: It produces vectors of size 384 (not 784).
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded successfully!\n")

    # -------------------------------------------------------------------------
    # 2. Inspecting a Single Embedding Vector
    # -------------------------------------------------------------------------
    sample_text = "This is a sample sentence to examine its embedding vector."
    sample_embedding = model.encode(sample_text)

    print("=" * 60)
    print("DEMO 1: Inspecting an Embedding Vector")
    print("=" * 60)
    print(f"Input text   : \"{sample_text}\"")
    print(f"Vector shape : {sample_embedding.shape}  -> 384 numbers representing this text")
    print(f"Vector type  : {type(sample_embedding)}")
    print(f"First 5 dims : {sample_embedding[:5]} ...\n")

    # -------------------------------------------------------------------------
    # 3. Comparing Semantically Similar Sentences
    # -------------------------------------------------------------------------
    t1_similar = "Hello, how are you?"
    t2_similar = "How are you doing today?"

    emb1_sim = model.encode(t1_similar)
    emb2_sim = model.encode(t2_similar)
    sim_score = cosine_similarity(emb1_sim, emb2_sim)

    print("=" * 60)
    print("DEMO 2: Semantically Similar Sentences")
    print("=" * 60)
    print(f"Sentence 1 : \"{t1_similar}\"")
    print(f"Sentence 2 : \"{t2_similar}\"")
    print(f"Score      : {sim_score:.4f}")
    print(f"Meaning    : {interpret_similarity(sim_score)}\n")

    # -------------------------------------------------------------------------
    # 4. Comparing Semantically Unrelated Sentences
    # -------------------------------------------------------------------------
    t1_diff = "Hello, how are you?"
    t2_diff = "Cats are very playful animals."

    emb1_diff = model.encode(t1_diff)
    emb2_diff = model.encode(t2_diff)
    diff_score = cosine_similarity(emb1_diff, emb2_diff)

    print("=" * 60)
    print("DEMO 3: Semantically Unrelated Sentences")
    print("=" * 60)
    print(f"Sentence 1 : \"{t1_diff}\"")
    print(f"Sentence 2 : \"{t2_diff}\"")
    print(f"Score      : {diff_score:.4f}")
    print(f"Meaning    : {interpret_similarity(diff_score)}\n")

    # -------------------------------------------------------------------------
    # 5. Comparing Different Words with Same Meaning (Synonyms & Paraphrasing)
    # -------------------------------------------------------------------------
    t1_paraphrase = "The weather outside is wonderful and sunny."
    t2_paraphrase = "It is a beautiful and bright day outdoors."

    emb1_para = model.encode(t1_paraphrase)
    emb2_para = model.encode(t2_paraphrase)
    para_score = cosine_similarity(emb1_para, emb2_para)

    print("=" * 60)
    print("DEMO 4: Paraphrased Sentences (Different words, same meaning)")
    print("=" * 60)
    print(f"Sentence 1 : \"{t1_paraphrase}\"")
    print(f"Sentence 2 : \"{t2_paraphrase}\"")
    print(f"Score      : {para_score:.4f}")
    print(f"Meaning    : {interpret_similarity(para_score)}\n")


if __name__ == "__main__":
    main()
