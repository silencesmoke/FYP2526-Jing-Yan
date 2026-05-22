import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dt_clean.csv", encoding="ISO-8859-1")

df["length"] = df["cleaned_text"].apply(lambda x: len(str(x).split()))

mean_len = np.mean(df["length"])
median_len = np.median(df["length"])
min_len = np.min(df["length"])
max_len = np.max(df["length"])
std_len = np.std(df["length"])

print("=== Abstract Length Statistics (Words) ===")
print(f"Mean: {mean_len:.2f}")
print(f"Median: {median_len:.2f}")
print(f"Min: {min_len}")
print(f"Max: {max_len}")
print(f"Std: {std_len:.2f}")

plt.figure(figsize=(10, 5))
plt.hist(df["length"], bins=40, color="#5a9bd4", edgecolor="black", alpha=0.8)
plt.title("Distribution of Abstract Word Count", fontsize=13, pad=10)
plt.xlabel("Number of Words in Abstract", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("length_dist.png", dpi=300)
plt.close()