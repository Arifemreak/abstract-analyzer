import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import os
import csv

# ---------- VERİ VE MODEL YÜKLEME ----------

df_all = pd.read_csv("makale_ozetleri_tumlesik.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df_all["Abstract"].tolist())

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_all["KMeans_Cluster"] = kmeans.fit_predict(embeddings)

# ML Sınıflandırıcı
X = embeddings
y = df_all["KMeans_Cluster"]
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Küme merkezleri
cluster_centroids = {}
for cluster_id in df_all["KMeans_Cluster"].unique():
    idxs = df_all[df_all["KMeans_Cluster"] == cluster_id].index
    cluster_centroids[cluster_id] = np.mean([embeddings[i] for i in idxs], axis=0)

cluster_descriptions = {
    0: "🧪 Mixed research content — omics, clustering, data fusion.",
    1: "🤖 AI in oncology — deep learning, imaging, decision support.",
    2: "🧬 Epigenetic regulation — DNA methylation, chromatin, histone mods."
}

# ---------- ABSTRACT ANALYSIS FONKSİYONU ----------

def analyze_abstract(input_abstract):
    try:
        input_vec = model.encode([input_abstract])[0]
        sims = cosine_similarity([input_vec], embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:3]
        top_matches = [(round(float(sims[i]), 3), df_all.iloc[i]["Title"]) for i in top_indices]

        assigned_cluster = clf.predict([input_vec])[0]
        centroid_vec = cluster_centroids[assigned_cluster]
        cluster_sim = float(cosine_similarity([input_vec], [centroid_vec])[0][0])

        description = cluster_descriptions.get(assigned_cluster, "Unknown cluster")
        warning = "⚠️ Low confidence in cluster assignment." if cluster_sim < 0.55 else ""

        return str(assigned_cluster), round(cluster_sim, 3), description, top_matches, warning

    except Exception as e:
        return "Error", 0, "Error", [], f"❌ {str(e)}"

# ---------- FEEDBACK LOGGING ----------

def log_feedback(input_text, predicted_cluster, user_feedback, correct_cluster=None):
    try:
        file_exists = os.path.isfile("user_feedback.csv")
        with open("user_feedback.csv", "a", newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Abstract", "Predicted Cluster", "User Feedback", "Correct Cluster"])
            writer.writerow([input_text, predicted_cluster, user_feedback, correct_cluster])
    except:
        pass

# ---------- PIPELINE ----------

def full_pipeline(input_text, user_feedback=None, correct_cluster=None):
    result = analyze_abstract(input_text)
    if user_feedback in ["Yes", "No"]:
        log_feedback(input_text, result[0], user_feedback, correct_cluster)
    return result

# ---------- GRADIO ARAYÜZÜ ----------

with gr.Blocks() as demo:
    gr.Markdown("## 🧬 Scientific Abstract Analyzer")
    gr.Markdown("Classify biomedical abstracts into clusters using AI + similarity scoring.")

    input_text = gr.Textbox(label="Paste abstract here", lines=4, placeholder="e.g. Aberrant DNA methylation plays a key role...")
    submit_btn = gr.Button("Analyze")

    with gr.Row():
        cluster_id = gr.Textbox(label="Predicted Cluster")
        similarity = gr.Textbox(label="Similarity to Cluster Center")

    description = gr.Textbox(label="Cluster Description", lines=2)
    warning = gr.Textbox(label="⚠️ Warning", interactive=False)

    similar_articles = gr.Dataframe(
        label="Top Similar Articles",
        headers=["Similarity", "Title"],
        datatype=["number", "str"],
        interactive=False
    )

    gr.Markdown("### 💬 Feedback")
    feedback_choice = gr.Radio(choices=["Yes", "No"], label="Was this prediction correct?")
    correct_cluster = gr.Textbox(label="If 'No', what is the correct cluster? (0/1/2)")
    feedback_btn = gr.Button("Submit Feedback")

    submit_btn.click(fn=full_pipeline, inputs=[input_text], outputs=[cluster_id, similarity, description, similar_articles, warning])
    feedback_btn.click(fn=full_pipeline, inputs=[input_text, feedback_choice, correct_cluster], outputs=[cluster_id, similarity, description, similar_articles, warning])

demo.launch()
