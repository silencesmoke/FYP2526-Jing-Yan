import os
import json
import re
import torch
import pandas as pd
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/gpt-neox-20b"
INPUT_FILE = "scopus_data.csv"
OUTPUT_FILE = "gpt_neox_20b_esrc_results.json"
TEMPERATURE = 0.1
TOP_P = 0.9
DO_SAMPLE = True
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 300
TORCH_DTYPE = torch.float16
DEVICE_MAP = "auto"


hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    device_map=DEVICE_MAP
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def build_prompt(article_id, abstract, index_keywords, author_keywords):
    return f"""
Please act as an academic domain classifier and classify the following article on “Applications of Digital Twin in Social Sciences” according to the ESRC (Economic and Social Research Council) discipline classification system.

Input Article Information:
Article ID: {article_id}
Abstract: {abstract}
Index Keywords: {index_keywords}
Author Keywords: {author_keywords}

Classification Task
Identify which ONE ESRC social science discipline the article belongs to, based on the following definitions:

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
  "esrc_domain": "ESRC Discipline",
  "reason": "Brief explanation of why this discipline was chosen, summarizing elements in the article that reflect this domain"
}}

The esrc_domain must be chosen from the ESRC disciplines listed above only.
Return only valid JSON. Do not include any extra text.
""".strip()


def extract_json(text):
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def classify_article(article_id, abstract, index_keywords, author_keywords):
    prompt = build_prompt(
        article_id=article_id,
        abstract=abstract,
        index_keywords=index_keywords,
        author_keywords=author_keywords
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    output_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return output_text


def get_runtime_info():
    return {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": DO_SAMPLE,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "torch_dtype": "float16",
        "device_map": DEVICE_MAP,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
    }


def main():
    df = pd.read_csv(
        INPUT_FILE,
        encoding="ISO-8859-1",
        on_bad_lines="skip"
    )


    runtime_info = get_runtime_info()
    print(json.dumps(runtime_info, indent=2, ensure_ascii=False))

    results = []

    for idx, row in df.iterrows():
        article_id = safe_text(row["article_id"]) if "article_id" in df.columns else str(idx + 1)
        abstract = safe_text(row.get("Abstract", ""))
        index_keywords = safe_text(row.get("Index Keywords", ""))
        author_keywords = safe_text(row.get("Author Keywords", ""))

        raw_output = classify_article(
            article_id=article_id,
            abstract=abstract,
            index_keywords=index_keywords,
            author_keywords=author_keywords
        )

        try:
            parsed_output = extract_json(raw_output)
        except Exception:
            parsed_output = {
                "article_id": article_id,
                "esrc_domain": "PARSE_ERROR",
                "reason": raw_output
            }

        print(parsed_output)

        results.append(parsed_output)

    output = {
        "runtime_info": runtime_info,
        "results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()