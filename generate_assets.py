import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, classification_report

# =================================================
# SETUP
# =================================================
os.makedirs("static/images", exist_ok=True)

# Load data & model
df = pd.read_csv("fingili.csv")
model = joblib.load("svm_sentiment_pipeline.pkl")

TEXT_COL = "kalimat"
LABEL_COL = "label tranformers"

X = df[TEXT_COL].astype(str).values
y_true = df[LABEL_COL].values
y_pred = model.predict(X)

labels = sorted(set(y_true))

# =================================================
# 1. SENTIMENT DISTRIBUTION
# =================================================
counts = df[LABEL_COL].value_counts()

plt.figure(figsize=(5, 4))
counts.plot(kind="bar")
plt.title("Distribusi Sentimen")
plt.ylabel("Jumlah")
plt.xlabel("Sentimen")
plt.tight_layout()
plt.savefig("static/images/sentiment_dist.png")
plt.close()

# =================================================
# 2. CONFUSION MATRIX
# =================================================
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("static/images/cm.png")
plt.close()

# =================================================
# 3. CLASSIFICATION REPORT (IMAGE)
# =================================================
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)

plt.figure(figsize=(10, 4))
plt.axis("off")

table = plt.table(
    cellText=report_df.values,
    colLabels=report_df.columns,
    rowLabels=report_df.index,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.4)

plt.title("Classification Report", pad=20)
plt.savefig("static/images/classification_report.png", bbox_inches="tight")
plt.close()

# =================================================
# 4. WORDCLOUD - SEMUA ULASAN
# =================================================
text_all = " ".join(df[TEXT_COL].astype(str))

wc_all = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text_all)

img_all = wc_all.to_image()
plt.figure(figsize=(8, 4))
plt.imshow(np.array(img_all))
plt.axis("off")
plt.tight_layout()
plt.savefig("static/images/wc_all.png")
plt.close()

# =================================================
# 5. WORDCLOUD PER SENTIMEN
# =================================================
for label in labels:
    subset = df[df[LABEL_COL] == label][TEXT_COL].astype(str)
    text_label = " ".join(subset)

    if text_label.strip() == "":
        continue

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text_label)

    img = wc.to_image()
    plt.figure(figsize=(8, 4))
    plt.imshow(np.array(img))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"static/images/wc_{label.lower()}.png")
    plt.close()

# =================================================
print("SEMUA ASSET BERHASIL DIBUAT DENGAN SUKSES.")
