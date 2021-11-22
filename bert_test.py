from sentence_transformers import SentenceTransformer, util
import scipy

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [line.strip() for line in open('data/similarity/user.txt')]
# Each sentence is encoded as a 1-D vector with 78 columns
sentence_embeddings = model.encode(sentences)

#@title Sematic Search Form

# code adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.py

query = 'Nobody has sane thoughts' #@param {type: 'string'}

queries = [query]
query_embeddings = model.encode(queries)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
number_top_matches = 5 #@param {type: "number"}

print("Semantic Search Results")

for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:number_top_matches]:
        print(sentences[idx].strip(), "(Cosine Score: %.4f)" % (1-distance))