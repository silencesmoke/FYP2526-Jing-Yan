import pandas as pd
import re
from nltk.corpus import stopwords

def main():
    df = pd.read_csv("dt_selected_6576_documents.csv", encoding="ISO-8859-1", on_bad_lines="skip")

    stop_words = set(stopwords.words("english"))

    def clean(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        return " ".join(words)

    df["cleaned_text"] = (
        df["Title"].fillna("").astype(str) + " " +
        df["Abstract"].fillna("").astype(str) + " " +
        df["Author Keywords"].fillna("").astype(str) + " " +
        df["Index Keywords"].fillna("").astype(str)
    )

    df["cleaned_text"] = df["cleaned_text"].apply(clean)
    df = df[df["cleaned_text"].str.strip() != ""]
    df.to_csv("dt_clean.csv", index=False, encoding="utf-8-sig")
    print(f"Clean finished. Output: dt_clean.csv")

if __name__ == "__main__":
    main()
