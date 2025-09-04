from sentence_transformers import SentenceTransformer, util
# from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")


def cos_sim(a, b):
    emb1 = model.encode(a)
    emb2 = model.encode(b)
    return util.cos_sim(emb1, emb2).item()


str1 = input("Embedding 1: ")
str2 = input("Embedding 2: ")
print("Embedding 1 vs Embedding 2:", cos_sim(str1, str2))
