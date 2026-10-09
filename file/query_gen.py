import math
import chromadb
print("loading spacy ...")
import spacy
print("loading KeyBERT ...")
from keybert import KeyBERT
# print("loading Retriever ...")
# import retriever

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT(model="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="documents"
)

def generate_queries(question: str, max_l: int) -> list:
    queries = set()
    max_limit ,min_limit = max_l, max_l
    print(max_limit,min_limit,sep=" ",end="/n")

    top_n = max(1, int(1 + math.log(max_l, 2)))
    min_limit = max(1, int(math.log(max_l + 1, 2)))
    max_limit = max(min_limit, int(math.log(max_l + 2, 1.5)))
   
    keywords = kw_model.extract_keywords(
        question,
        keyphrase_ngram_range = (min_limit, max_limit),
        stop_words = "english",
        top_n = top_n - 1,
    )
    for kw, _ in keywords:
        queries.add(kw.strip())
    queries.add(question)
    return list(queries)

def filter_tokens(question: str):
    size = len(question.split())
    doc = nlp(question)

    filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
    max_l = len(filtered_tokens)
    final_tokens = ""

    for i in filtered_tokens:
        final_tokens+=f"{i} "
    final_tokens = final_tokens.strip()
    return final_tokens, max_l

if __name__ == "__main__":
    question = "what are the reasons for persistant dry cough and low blood pressure"
    print("=========================================================================")
    while True:
        print(":::Question:::\n",question)
        size = len(question.split())
        doc = nlp(question)

        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        max_l = len(filtered_tokens)
        final_tokens = ""

        for i in filtered_tokens:
            final_tokens+=f"{i} "
        final_tokens = final_tokens.strip()
        print(final_tokens)
        queries = [final_tokens,]
        if max_l > 4:
            queries = generate_queries(question=final_tokens, max_l = max_l)
            print("\n\n:::Queries:::")
            num = 1
            for i in queries:
                print(num,") ",i)
                num+=1
        print("\n:::responses:::")
        num = 1
        for i in queries:
            results = collection.query(
            query_texts=i,
                n_results=2                )
            print(num,") Query: ",i)
            print(results["documents"])
            # retriever_object = retriever.ChromaRetriever()
            
            # for i in queries:
            #     response = retriever_object.retrieve(query=i)
            #     print(response,end="\n\n")
        print("=========================================================================")
        question = input(":::Enter question => ")


"""
What is the difference between type 1 and type 2 diabetes definitions and what is the hemoglobin A1c target for diabetic patients 
How does coronary artery disease develop from plaque buildup and what are the specific arterial branches that supply blood to the heart muscle

difference between valvular stenosis and regurgitation heart valve disease
normal adult blood pressure targets according to Hypertension Canada

SCAD vs plaque heart attack
smoking cessation heart timeline
"""