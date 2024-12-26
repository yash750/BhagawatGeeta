from flask import Flask, render_template, request, jsonify
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    pipeline,
    LlamaForCausalLM,
    LlamaTokenizer,
)
import os

app = Flask(__name__)

# Load SBERT model for embeddings
sbert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
sbert_tokenizer = AutoTokenizer.from_pretrained(sbert_model_name)
sbert_model = AutoModel.from_pretrained(sbert_model_name)

# Load Llama model for text generation
llama_model_name = "your-llama-model-path"
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)
llama_model = LlamaForCausalLM.from_pretrained(
    llama_model_name, torch_dtype=torch.float16, device_map="auto"
)

# Load your dataset embeddings and text
data_embeddings = torch.load("data/embeddings.pt")  # Precomputed embeddings
data_texts = open("data/dataset_texts.txt").read().splitlines()  # Corresponding text


# Cosine similarity function
def cosine_similarity(a, b):
    return (a @ b.T) / (torch.norm(a) * torch.norm(b))


# Retrieval function
def retrieve_context(query, k=3):
    query_embedding = sbert_model(**sbert_tokenizer(query, return_tensors="pt"))[
        0
    ].mean(dim=1)
    similarities = [
        cosine_similarity(query_embedding, data_embedding).item()
        for data_embedding in data_embeddings
    ]
    top_k_indices = sorted(
        range(len(similarities)), key=lambda i: similarities[i], reverse=True
    )[:k]
    top_k_texts = [data_texts[i] for i in top_k_indices]
    return top_k_texts


# Generation function
def generate_answer(context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = llama_tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = llama_model.generate(**inputs, max_length=150, num_beams=5)
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.form.get("query")

    # Step 1: Retrieve context
    retrieved_contexts = retrieve_context(user_query, k=3)
    context = " ".join(retrieved_contexts)

    # Step 2: Generate answer
    answer = generate_answer(context, user_query)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(port=5002)
