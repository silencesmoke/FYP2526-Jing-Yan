from huggingface_hub import login

login("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")

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import json


model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def build_prompt(article_id, title, cleaned_abstract, index_keyword, author_keyword):
    return f"""
Please act as an academic domain classifier and classify the following article on “Applications of Digital Twin in Social Sciences” according to the ESRC (Economic and Social Research Council) discipline classification system.

Input Article Information:
Article ID: {article_id}
Title: {title}
Cleaned Abstract: {cleaned_abstract}
Index Keywords: {index_keyword}
Author Keywords: {author_keyword}

Classification Task
Identify which ESRC social science discipline(s) the article belongs to, based on the following definitions:

Demography
Demography is the study of populations and population changes and trends, using resources such as statistics of births, deaths, and disease.

Area and development studies
Area and development studies is a multidisciplinary branch of the social sciences which addresses a range of social and economic issues associated with low and middle-income countries in different geographical regions.

Economics
Economics seeks to understand how individuals interact within the social structure, to address key questions about the production and exchange of goods and services.

Economic and social history
Economic and social history looks at past events to learn from history and better understand the processes of contemporary or near contemporary society.

Education
Education is one of the most important social sciences, exploring how people learn and develop.

Environmental planning
Environmental planning explores the decision-making processes for managing relationships within and between human systems and natural systems, in order to manage these processes in an effective, transparent, and equitable manner.

Human geography
Human geography studies the world, its people, communities, and cultures, and differs from physical geography mainly in that it focuses on human activities and their impact, for instance on environmental change.

Linguistics
Linguistics focuses on how people communicate and create meaning through language. ESRC covers applied linguistics research in the areas of computational and corpus linguistics, psycholinguistics, sociolinguistics, discourse analysis, language acquisition and interdisciplinary social science research involving linguistics.

Management and business studies
Management and business studies explores a wide range of aspects relating to the activities and management of business, such as strategic and operational management, organisational psychology, employment relations, marketing, accounting, finance, logistics and productivity.

Politics and international studies
Politics focuses on democracy and the relationship between people and policy, at all levels up from the individual to a national and international level.
International studies is the study of relationships between countries, including the roles of other organisations.

Psychology
Psychology studies the human mind and behaviour to try to understand how people and groups experience the world through various emotions, ideas, cognitive processes, and conscious states. ESRC also covers elements of mental and public health.

Science and technology studies
Science and technology studies is concerned with what scientists do, what their role is in our society, the history and culture of science, and the policies and debates that shape our modern scientific and technological world.

Social anthropology
Social anthropology is the study of how human societies and social structures are organised and understood.

Social policy
Social policy is an interdisciplinary and applied subject concerned with the analysis of societies’ responses to social need, focusing on aspects of society, economy, public and global health, and policy that are necessary to human existence, and how these can be provided.

Social work
Social work focuses on social change, problem solving in human relationships and the empowerment and liberation of people to enhance social justice. ESRC also covers broader social care for adults and children.

Social statistics, methods, and computing
Social statistics, methods and computing involves the collection and analysis of quantitative and qualitative social science data.

Socio-legal studies
Socio-legal studies focuses on the social, political, and economic influences and its impact on the law and the legal system.

Sociology
Sociology involves groups of people, rather than individuals, and attempts to understand the way people relate to each other and function as a society or social subgroups.

Briefly explain the reason for the classification by extracting the elements in the article that reflect the characteristics of the identified domain.

Output format:
{{
  "article_id": "{article_id}",
  "abstract_snippet": "Brief explanation of why this discipline was chosen, summarizing elements in the article that reflect this domain",
  "flan_t5_large_prediction": "ESRC Discipline"
}}

Return only valid JSON. Do not return any other text.
""".strip()


def classify_flan_t5(article_id, title, cleaned_abstract, index_keyword, author_keyword):
    prompt = build_prompt(
        article_id,
        title,
        cleaned_abstract,
        index_keyword,
        author_keyword
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.01,
        do_sample=False
    )

    res = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return res


df = pd.read_csv("scopus_data.csv", encoding="gbk")

results = []

for _, row in df.iterrows():
    aid = row["article_id"]
    title = str(row["Title"])
    cleaned_abstract = str(row["cleaned_abstract"])
    index_keyword = str(row["Index Keywords"])
    author_keyword = str(row["Author Keywords"])

    pred = classify_flan_t5(
        aid,
        title,
        cleaned_abstract,
        index_keyword,
        author_keyword
    )

    print(pred)

    try:
        pred_json = json.loads(pred)
        results.append(pred_json)

    except:
        results.append({
            "article_id": int(aid),
            "raw_output": pred
        })


with open("esrc_flan_t5_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Finished.")
