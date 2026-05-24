import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

JUDGE_MODEL = "Qwen/Qwen2-7B-Instruct"

CSV_PATH = "scopus_data.csv"
FLAN_RESULT_PATH = "esrc_flan_t5_result.json"
NEOX_RESULT_PATH = "gpt_neox_20b_esrc_results.json"

OUTPUT_PATH = "esrc_judge_results.json"
TEST_NUM = 10

print("Loading judge model...")

tokenizer = AutoTokenizer.from_pretrained(
    JUDGE_MODEL,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

print("Judge model loaded.\n")

PROMPT_TEMPLATE = """
Prompt 2
You are an academic annotation evaluation judge. Your task is to evaluate and compare the ESRC domain classification results produced by Flan-T5-large and GPT-neoX-20B about applications of Digital Twin in the Social Sciences.

Please perform a step-by-step reasoning and assess each model’s annotation quality according to the following five criteria:

Accuracy – Does the annotated ESRC domain precisely match the article’s thematic focus and disciplinary scope?
Reasonableness – Are the justifications logically coherent, well-structured, and supported by clear evidence from the article?
Consistency – Across similar articles or related subfields, does the model maintain stable and uniform domain labeling?
Relevance – How well do the identified domains and reasoning align with the central research context?
Coherence – Is the reasoning presented in a clear, connected, and academically sound manner?

You must provide detailed analytical comments for each model and then determine the overall winner based on comparative evaluation.

Input to Evaluate:

Article ID:
{article_id}

Article Abstract:
{abstract}

Flan-T5-large annotation:
{flan_annotation}

GPT-neoX-20B annotation:
{neox_annotation}

Output Format (JSON):
{{
  "evaluator": "{judge_model}",
  "assessment": [
    {{
      "model": "Flan-T5-Large",
      "accuracy_score": "1-5",
      "reasonableness_score": "1-5",
      "consistency_score": "1-5",
      "relevance_score": "1-5",
      "coherence_score": "1-5",
      "comments": "Comprehensive qualitative assessment, including strengths, weaknesses, and improvement suggestions."
    }},
    {{
      "model": "GPT-neoX-20B",
      "accuracy_score": "1-5",
      "reasonableness_score": "1-5",
      "consistency_score": "1-5",
      "relevance_score": "1-5",
      "coherence_score": "1-5",
      "comments": "Comprehensive qualitative assessment, including strengths, weaknesses, and improvement suggestions."
    }}
  ],
  "conclusion": "A comparative summary of the two models’ overall performance with justification and the final declared winner."
}}

Important:
- Return JSON only.
- Do not add markdown, explanations, or text outside the JSON.
"""

def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)

    if not match:
        raise ValueError("No JSON object found in model output.")

    json_text = match.group(0)
    return json.loads(json_text)


def generate_judgement(prompt):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    result_text = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return result_text


def get_field(item, possible_keys):
    for key in possible_keys:
        if key in item:
            return item[key]
    return ""


if __name__ == "__main__":

    df = pd.read_csv(CSV_PATH)

    df["article_id"] = df["article_id"].astype(str)

    abstract_dict = dict(
        zip(
            df["article_id"],
            df["Abstract"]
        )
    )

    with open(FLAN_RESULT_PATH, "r", encoding="utf-8") as f:
        flan_data = json.load(f)

    with open(NEOX_RESULT_PATH, "r", encoding="utf-8") as f:
        neox_data = json.load(f)

    flan_dict = {
        str(item["article_id"]): item
        for item in flan_data
    }

    neox_dict = {
        str(item["article_id"]): item
        for item in neox_data
    }

    common_ids = list(
        set(flan_dict.keys()) & set(neox_dict.keys()) & set(abstract_dict.keys())
    )

    common_ids = common_ids[:TEST_NUM]

    results = []

    for idx, article_id in enumerate(common_ids, 1):

        print(f"Processing {idx}/{len(common_ids)} | Article ID: {article_id}")

        flan_item = flan_dict[article_id]
        neox_item = neox_dict[article_id]

        abstract = abstract_dict.get(article_id, "")

        flan_annotation = get_field(
            flan_item,
            [
                "flan_t5_large_prediction",
                "prediction",
                "annotation",
                "result"
            ]
        )

        neox_annotation = get_field(
            neox_item,
            [
                "gpt_neox_20b_prediction",
                "neox_prediction",
                "prediction",
                "annotation",
                "result"
            ]
        )

        prompt = PROMPT_TEMPLATE.format(
            judge_model=JUDGE_MODEL,
            article_id=article_id,
            abstract=abstract[:2500],
            flan_annotation=flan_annotation,
            neox_annotation=neox_annotation
        )

        raw_output = generate_judgement(prompt)

        try:
            result_json = extract_json(raw_output)
        except Exception as e:
            result_json = {
                "evaluator": JUDGE_MODEL,
                "article_id": article_id,
                "error": str(e),
                "raw_output": raw_output
            }

        results.append(result_json)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nFinished.")
    print(f"Saved to: {OUTPUT_PATH}")
