import json
import re
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = r"D:\models\Qwen/Qwen2-7B-Instruct"

FLAN_RESULT_PATH = "esrc_flan_t5_result.json"
NEOX_RESULT_PATH = "gpt_neox_20b_esrc_results.json"

OUTPUT_PATH = "judge_flan_vs_neox_results.json"
FINAL_OUTPUT_PATH = "final_filtered_esrc_annotations.json"
FINAL_CSV_PATH = "final_filtered_esrc_annotations.csv"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

model.eval()

PROMPT_TEMPLATE = """
You are an academic annotation evaluation judge.

Your task is to evaluate and compare two ESRC discipline classification results about applications of Digital Twin in the Social Sciences.

Assessment criteria:
Accuracy – Does the ESRC domain match the article’s main thematic focus?
Reasonableness – Is the selected ESRC domain a plausible disciplinary interpretation?
Consistency – Is the label consistent with how similar Digital Twin/social science articles should normally be classified?
Relevance – Is the label relevant to the central research context?
Coherence – Is the disciplinary choice coherent with the article’s topic, method, and application context?

Article ID:
{article_id}

Article abstract snippet:
{abstract_snippet}

Flan-T5-large annotation:
Label: {flan_label}

GPT-NeoX-20B annotation:
Label: {neox_label}
Reason: {neox_reason}

Return JSON only in this exact format:
{{
  "article_id": "{article_id}",
  "flan_t5_large_label": "{flan_label}",
  "gpt_neox_20b_label": "{neox_label}",
  "assessment": [
    {{
      "model": "Flan-T5-Large",
      "accuracy_score": "<integer from 1 to 5>",
      "reasonableness_score": "<integer from 1 to 5>",
      "consistency_score": "<integer from 1 to 5>",
      "relevance_score": "<integer from 1 to 5>",
      "coherence_score": "<integer from 1 to 5>",
      "total_score": "<sum of the five scores>",
      "comments": "Brief qualitative assessment of whether the label is supported by the abstract snippet."
    }},
    {{
      "model": "GPT-NeoX-20B",
      "accuracy_score": "<integer from 1 to 5>",
      "reasonableness_score": "<integer from 1 to 5>",
      "consistency_score": "<integer from 1 to 5>",
      "relevance_score": "<integer from 1 to 5>",
      "coherence_score": "<integer from 1 to 5>",
      "total_score": "<sum of the five scores>",
      "comments": "Brief qualitative assessment of whether the label and reason are supported by the abstract snippet."
    }}
  ],
  "winner": "Flan-T5-Large / GPT-NeoX-20B / Tie",
  "final_esrc_domain": "The final retained ESRC label, or empty string if filtered out",
  "retained": true,
  "filter_reason": "",
  "conclusion": "Brief explanation of the final decision."
}}

Scoring rule:
- 5 = clearly correct and strongly supported by the article evidence
- 4 = mostly appropriate, with minor ambiguity
- 3 = acceptable but somewhat broad or ambiguous
- 2 = weakly related to the article evidence
- 1 = incorrect or unsupported

Filtering rule:
- Select the model with the higher total score as the winner.
- If both models have the same total score, choose Tie and select the more suitable ESRC label based on the abstract snippet.
- If the selected final label has any individual score lower than 3, set retained to false.
- If the selected final label has total_score lower than 15, set retained to false.
- If retained is false, set final_esrc_domain to an empty string and explain the reason in filter_reason.
- Scores must be integers from 1 to 5.
- total_score must equal the sum of the five individual scores.
- Do not copy placeholder values.
- Return JSON only. No markdown. No extra text.
"""

def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found.")
    return json.loads(match.group(0))

def call_judge(prompt):
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
        [text],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    output_ids = generated_ids[0][inputs.input_ids.shape[-1]:]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return output_text.strip()

def get_field(item, key):
    value = item.get(key, "")
    if value is None:
        return ""
    return str(value).strip()

def score_to_int(value):
    try:
        return int(value)
    except Exception:
        return 0

def compute_total(assessment_item):
    score_keys = [
        "accuracy_score",
        "reasonableness_score",
        "consistency_score",
        "relevance_score",
        "coherence_score"
    ]
    total = sum(score_to_int(assessment_item.get(k, 0)) for k in score_keys)
    assessment_item["total_score"] = total
    return total

def apply_filter_and_fix(result_json):
    assessments = result_json.get("assessment", [])

    if len(assessments) != 2:
        result_json["retained"] = False
        result_json["final_esrc_domain"] = ""
        result_json["filter_reason"] = "Invalid assessment structure"
        return result_json

    for item in assessments:
        compute_total(item)

    flan_assessment = assessments[0]
    neox_assessment = assessments[1]

    flan_total = flan_assessment["total_score"]
    neox_total = neox_assessment["total_score"]

    flan_label = result_json.get("flan_t5_large_label", "")
    neox_label = result_json.get("gpt_neox_20b_label", "")

    winner = result_json.get("winner", "")

    if winner not in ["Flan-T5-Large", "GPT-NeoX-20B", "Tie"]:
        if flan_total > neox_total:
            winner = "Flan-T5-Large"
        elif neox_total > flan_total:
            winner = "GPT-NeoX-20B"
        else:
            winner = "Tie"

    result_json["winner"] = winner

    if winner == "Flan-T5-Large":
        selected_assessment = flan_assessment
        selected_label = flan_label
    elif winner == "GPT-NeoX-20B":
        selected_assessment = neox_assessment
        selected_label = neox_label
    else:
        selected_label = result_json.get("final_esrc_domain", "").strip()
        if selected_label == "":
            selected_label = neox_label
        selected_assessment = flan_assessment if flan_total >= neox_total else neox_assessment

    score_keys = [
        "accuracy_score",
        "reasonableness_score",
        "consistency_score",
        "relevance_score",
        "coherence_score"
    ]

    individual_scores = [
        score_to_int(selected_assessment.get(k, 0))
        for k in score_keys
    ]

    selected_total = score_to_int(selected_assessment.get("total_score", 0))

    if any(score < 3 for score in individual_scores):
        result_json["retained"] = False
        result_json["final_esrc_domain"] = ""
        result_json["filter_reason"] = "At least one individual score is lower than 3"
    elif selected_total < 15:
        result_json["retained"] = False
        result_json["final_esrc_domain"] = ""
        result_json["filter_reason"] = "Total score is lower than 15"
    else:
        result_json["retained"] = True
        result_json["final_esrc_domain"] = selected_label
        result_json["filter_reason"] = ""

    return result_json

if __name__ == "__main__":

    with open(FLAN_RESULT_PATH, "r", encoding="utf-8") as f:
        flan_data = json.load(f)

    with open(NEOX_RESULT_PATH, "r", encoding="utf-8") as f:
        neox_raw = json.load(f)

    neox_data = neox_raw["results"]

    flan_dict = {
        str(item["article_id"]).strip(): item
        for item in flan_data
    }

    neox_dict = {
        str(item["article_id"]).strip(): item
        for item in neox_data
    }

    common_ids = sorted(
        set(flan_dict.keys()) & set(neox_dict.keys()),
        key=lambda x: int(x)
    )

    print("Flan count:", len(flan_dict))
    print("NeoX count:", len(neox_dict))
    print("Matched count:", len(common_ids))

    results = []

    for idx, article_id in enumerate(common_ids, 1):
        print(f"Processing {idx}/{len(common_ids)} | Article ID: {article_id}")

        flan_item = flan_dict[article_id]
        neox_item = neox_dict[article_id]

        abstract_snippet = get_field(flan_item, "abstract_snippet")
        flan_label = get_field(flan_item, "flan_t5_large_prediction")

        neox_label = get_field(neox_item, "esrc_domain")
        neox_reason = get_field(neox_item, "reason")

        prompt = PROMPT_TEMPLATE.format(
            article_id=article_id,
            abstract_snippet=abstract_snippet[:2500],
            flan_label=flan_label,
            neox_label=neox_label,
            neox_reason=neox_reason[:1200]
        )

        try:
            raw_output = call_judge(prompt)
            result_json = extract_json(raw_output)

            result_json["raw_output"] = raw_output
            result_json["article_id"] = article_id
            result_json["abstract_snippet"] = abstract_snippet
            result_json["flan_t5_large_label"] = flan_label
            result_json["gpt_neox_20b_label"] = neox_label
            result_json["gpt_neox_20b_reason"] = neox_reason

            result_json = apply_filter_and_fix(result_json)

        except Exception as e:
            result_json = {
                "article_id": article_id,
                "abstract_snippet": abstract_snippet,
                "flan_t5_large_label": flan_label,
                "gpt_neox_20b_label": neox_label,
                "gpt_neox_20b_reason": neox_reason,
                "assessment": [],
                "winner": "",
                "final_esrc_domain": "",
                "retained": False,
                "filter_reason": str(e),
                "raw_output": ""
            }

        results.append(result_json)

        if idx % 20 == 0:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(0.2)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    final_results = []

    for item in results:
        if item.get("retained") is True:
            final_results.append({
                "article_id": item.get("article_id", ""),
                "final_esrc_domain": item.get("final_esrc_domain", ""),
                "winner": item.get("winner", ""),
                "flan_t5_large_label": item.get("flan_t5_large_label", ""),
                "gpt_neox_20b_label": item.get("gpt_neox_20b_label", "")
            })

    with open(FINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    pd.DataFrame(final_results).to_csv(
        FINAL_CSV_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Finished.")
    print(f"Full judge results saved to: {OUTPUT_PATH}")
    print(f"Final filtered annotations saved to: {FINAL_OUTPUT_PATH}")
    print(f"Final filtered CSV saved to: {FINAL_CSV_PATH}")
    print(f"Retained articles: {len(final_results)}")