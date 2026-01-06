import pandas as pd

df = pd.read_csv("data/dataset/melb_data.csv")
df["Suburb"] = df["Suburb"].str.strip()

TARGET_SUBURBS = ["Melbourne", "Camberwell", "Toorak"]
N_PER_SUBURB = 60

subsets = []

for suburb in TARGET_SUBURBS:
    df_suburb = df[df["Suburb"] == suburb]
    df_suburb = df_suburb.dropna(subset=["Price"])

    available = len(df_suburb)
    n = min(N_PER_SUBURB, available)

    print(f"{suburb}: available={available}, taking={n}")

    df_sample = df_suburb.sample(n=n, random_state=42)
    subsets.append(df_sample)

df_final = pd.concat(subsets, ignore_index=True)
df_final.to_csv("data/melbourne_subset.csv", index=False)

print("\n✅ Final dataset shape:", df_final.shape)
print(df_final["Suburb"].value_counts())
