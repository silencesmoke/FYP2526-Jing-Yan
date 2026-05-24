import json
import pandas as pd


def json_to_excel(input_json, output_excel):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    df.to_excel(output_excel, index=False)

    print(f"Conversion completed: {output_excel}")
    print(df.head())


json_to_excel(
    input_json="esrc_flan_t5_result.json",
    output_excel="esrc_flan_t5_result_clean.xlsx"
)

json_to_excel(
    input_json="gpt_neox_20b_esrc_results.json",
    output_excel="gpt_neox_20b_esrc_results.json"
)
