import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# ===================== 停用词 =====================
nltk.download('stopwords', quiet=True)
stop_words = stopwords.words('english')

domain_stopwords = [
    'digital', 'twin', 'twins',
    'technology', 'technologies',
    'study', 'research', 'paper', 'article',
    'data', 'system', 'systems', 'model', 'models',
    'based', 'using', 'use'
]
stop_words.extend(domain_stopwords)

# ===================== 读取数据 =====================
df = pd.read_csv("dt_selected_6576_documents.csv", encoding="utf-8-sig")

# ===================== 预处理（仅 Abstract） =====================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    word_list = text.split()
    word_list = [w for w in word_list if w not in stop_words and len(w) > 2]
    return ' '.join(word_list)

if 'cleaned_abstract' not in df.columns:
    df['cleaned_abstract'] = df['Abstract'].apply(preprocess_text)

# ===================== 模型 =====================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.2,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=14,
    min_samples=2,
    cluster_selection_epsilon=0.3,
    prediction_data=True
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    nr_topics=9,
    min_topic_size=10,
    n_gram_range=(2, 3),
    low_memory=False,
    verbose=True
)

# ===================== 训练 =====================
topics, probabilities = topic_model.fit_transform(df['cleaned_abstract'].tolist())
df['assigned_topic'] = topics

# ===================== 自定义主题名称 =====================
topic_names = {}

for topic_id in sorted(topic_model.get_topic_info().Topic):
    if topic_id == -1:
        topic_names[topic_id] = "Outliers"
    else:
        topic_names[topic_id] = f"Topic {topic_id}"

topic_model.set_topic_labels(topic_names)

# ===================== 输出基础统计 =====================
print("\n=== TOPIC COUNT STATISTICS ===")
topic_info = topic_model.get_topic_info()
print(topic_info)

print("\n=== TOPIC KEYWORDS ===")
for topic_id in sorted(topic_info.Topic):
    if topic_id == -1:
        continue
    words = [word for word, score in topic_model.get_topic(topic_id)[:10]]
    print(f"Topic {topic_id}: {', '.join(words)}")

# ===================== Topic Table =====================
print("\nGenerating topic table...")

total_docs = len(df)

tokenized_docs = [doc.split() for doc in df['cleaned_abstract'].tolist()]
dictionary = Dictionary(tokenized_docs)

topic_table_rows = []

for topic_id in sorted(topic_info.Topic):
    topic_count = int(topic_info.loc[topic_info.Topic == topic_id, "Count"].values[0])
    percentage = topic_count / total_docs * 100

    if topic_id == -1:
        label = "Outliers"
        keywords = ""
        coherence_score = ""
    else:
        label = f"Topic {topic_id}"

        raw_topic_words = [word for word, score in topic_model.get_topic(topic_id)[:10]]
        keywords = ", ".join(raw_topic_words)

        coherence_words = []
        for phrase in raw_topic_words:
            for word in phrase.split():
                if word in dictionary.token2id and word not in coherence_words:
                    coherence_words.append(word)

        if len(coherence_words) >= 2:
            coherence_model = CoherenceModel(
                topics=[coherence_words],
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence='c_v',
                processes=1
            )
            coherence_score = coherence_model.get_coherence()
        else:
            coherence_score = ""

    topic_df = df[df['assigned_topic'] == topic_id].copy()

    if len(topic_df) > 0:
        if probabilities is not None:
            try:
                topic_df['topic_probability'] = probabilities[topic_df.index]
                topic_df = topic_df.sort_values('topic_probability', ascending=False)
            except Exception:
                pass

        if 'Title' in topic_df.columns:
            representative_titles = topic_df['Title'].dropna().astype(str).head(3).tolist()
        elif 'title' in topic_df.columns:
            representative_titles = topic_df['title'].dropna().astype(str).head(3).tolist()
        else:
            representative_titles = ["No title column found"]
    else:
        representative_titles = []

    representative_documents = " || ".join(representative_titles)

    topic_table_rows.append({
        "label": label,
        "keywords": keywords,
        "representative_documents": representative_documents,
        "document_count": topic_count,
        "percentage": round(percentage, 2),
        "coherence_score": round(coherence_score, 4) if coherence_score != "" else ""
    })

topic_table = pd.DataFrame(topic_table_rows)
topic_table.to_csv("dt_topic_table.csv", index=False, encoding="utf-8-sig")
print("Topic table saved: dt_topic_table.csv")

# ===================== Topic Quality Validation =====================
valid_topics = topic_table[
    (topic_table["label"] != "Outliers") &
    (topic_table["coherence_score"] != "")
].copy()

if len(valid_topics) > 0:
    mean_coherence = valid_topics["coherence_score"].astype(float).mean()
else:
    mean_coherence = ""

all_topic_words = []

for topic_id in sorted(topic_info.Topic):
    if topic_id == -1:
        continue
    words = [word for word, score in topic_model.get_topic(topic_id)[:10]]
    all_topic_words.extend(words)

if len(all_topic_words) > 0:
    topic_diversity = len(set(all_topic_words)) / len(all_topic_words)
else:
    topic_diversity = ""

if -1 in topic_info.Topic.values:
    number_of_outliers = int(topic_info.loc[topic_info.Topic == -1, "Count"].values[0])
else:
    number_of_outliers = 0

quality_summary = pd.DataFrame([{
    "mean_coherence_c_v": round(mean_coherence, 4) if mean_coherence != "" else "",
    "topic_diversity": round(topic_diversity, 4) if topic_diversity != "" else "",
    "number_of_topics": len(valid_topics),
    "number_of_documents": total_docs,
    "number_of_outliers": number_of_outliers
}])

quality_summary.to_csv("dt_topic_quality_summary.csv", index=False, encoding="utf-8-sig")
print("Topic quality summary saved: dt_topic_quality_summary.csv")
print(quality_summary)

# ===================== Manual Validation Samples =====================
manual_validation_rows = []

for topic_id in sorted(topic_info.Topic):
    if topic_id == -1:
        continue

    topic_df = df[df['assigned_topic'] == topic_id].copy()

    if len(topic_df) == 0:
        continue

    if probabilities is not None:
        try:
            topic_df['topic_probability'] = probabilities[topic_df.index]
            topic_df = topic_df.sort_values('topic_probability', ascending=False)
        except Exception:
            pass

    sample_cols = []

    for col in ['Title', 'title', 'Abstract', 'cleaned_abstract']:
        if col in topic_df.columns:
            sample_cols.append(col)

    sample_df = topic_df.head(5)[sample_cols].copy()
    sample_df.insert(0, "topic_id", topic_id)

    manual_validation_rows.append(sample_df)

if manual_validation_rows:
    manual_validation_table = pd.concat(manual_validation_rows, ignore_index=True)
    manual_validation_table.to_csv(
        "dt_manual_validation_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print("Manual validation samples saved: dt_manual_validation_samples.csv")

# ===================== 保存带主题编号的数据 =====================
df.to_csv("dt_documents_with_topics.csv", index=False, encoding="utf-8-sig")
print("Document-topic assignment saved: dt_documents_with_topics.csv")

# ===================== 画图 =====================
print("\nGenerating images...")

# 1 主题数量图
topic_counts = topic_model.get_topic_info().sort_values("Topic")

plt.figure(figsize=(10, 5))
plt.bar(topic_counts["Topic"].astype(str), topic_counts["Count"])
plt.title("Topic Count Distribution")
plt.xlabel("Topic ID")
plt.ylabel("Number of Documents")
plt.tight_layout()
plt.savefig("topic_count.png", dpi=300)
plt.close()

# 2 关键词条形图
topic_model.visualize_barchart(
    top_n_topics=9,
    n_words=10,
    custom_labels=True
).write_image("topic_keywords.png")

# 3 层级图
topic_model.visualize_hierarchy(
    custom_labels=True
).write_image("topic_hierarchy.png")

# 4 主题距离图
fig = topic_model.visualize_topics(
    top_n_topics=9,
    width=1200,
    height=900,
    custom_labels=True
)

fig.update_layout(
    sliders=[],
    showlegend=True
)

for trace in fig.data:
    if hasattr(trace, "mode") and trace.mode and "markers" in trace.mode:
        trace.update(
            mode="markers+text",
            text=[trace.name] * len(trace.x),
            textposition="top center",
            marker=dict(
                size=trace.marker.size,
                opacity=0.75,
                line=dict(width=1)
            )
        )

fig.write_image("intertopic_distance_map_labeled.png", scale=3)

# 5 文档 UMAP 图
fig_docs = topic_model.visualize_documents(
    df['cleaned_abstract'].tolist(),
    custom_labels=True,
    width=1100,
    height=800
)
fig_docs.write_image("document_umap_by_topic.png", scale=3)

# 6 主题相似度热力图
fig_heatmap = topic_model.visualize_heatmap(
    n_clusters=3,
    custom_labels=True
)
fig_heatmap.write_image("topic_similarity_heatmap.png", scale=3)

# 7 时间趋势 + 8 年份分布
try:
    topics_over_time = topic_model.topics_over_time(
        docs=df['cleaned_abstract'].tolist(),
        timestamps=df['Year'].tolist(),
        nr_bins=10
    )

    fig_time = topic_model.visualize_topics_over_time(
        topics_over_time,
        top_n_topics=9,
        custom_labels=True
    )
    fig_time.write_image("topics_over_time.png", scale=3)

    topics_per_class = topic_model.topics_per_class(
        docs=df['cleaned_abstract'].tolist(),
        classes=df['Year'].tolist()
    )

    fig_class = topic_model.visualize_topics_per_class(
        topics_per_class,
        top_n_topics=9,
        custom_labels=True
    )
    fig_class.write_image("topics_per_class_by_year.png", scale=3)

except Exception as e:
    print("Time charts skipped.")
    print(e)

print("All images and tables saved!")