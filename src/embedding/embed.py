import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_simalarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    return dot_product / (norm_emb1 * norm_emb2)

model = SentenceTransformer("all-MiniLM-L6-v2") #784 Vector Size
# text = "this is the first sentence"
# embedding = model.encode(text)
# print(embedding)
# print(embedding.shape)   # (784,)
# print(embedding.ndim)

#Similar
t1 = "Hello, how are you?"
t2 = "How are you doing today?"

emb1 = model.encode(t1)
emb2 = model.encode(t2)

print(cosine_simalarity(emb1, emb2))

# Different
t1 = "Hello, how are you?"
t2 = "Cats are very Playful Animal"

emb1 = model.encode(t1)
emb2 = model.encode(t2)

print(cosine_simalarity(emb1, emb2))
