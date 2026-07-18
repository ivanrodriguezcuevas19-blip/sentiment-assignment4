"""
Full sentiment analysis pipeline for Assignment #4.

  1. COLLECTION    - load review data from CSV (data/)
  2. PROCESSING    - run 3 sentiment analysis methods, compare accuracy against ground-truth star ratings
  3. INGESTION     - write results to a local SQLite database (normalized into two tables: reviews, sentiment_scores)
  4. VISUALIZATION - generate charts into outputs/
"""

import sqlite3
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/Reviews.csv"   
TEXT_COLUMN = "Text"                    
STARS_COLUMN = "Score"                  
SAMPLE_SIZE = 40000                     
USE_BERT = True

DB_PATH = "outputs/sentiment_analysis.db"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)



# 1. COLLECTION
def load_data():
    print("=" * 60)
    print("STEP 1: COLLECTION")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH)
    df = df[[TEXT_COLUMN, STARS_COLUMN]].dropna()
    df.columns = ["text", "stars"]
    df["text"] = df["text"].astype(str)

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)

    df = df.reset_index(drop=True)
    df["review_id"] = df.index + 1

    # Ground truth label derived from star rating
    df["true_sentiment"] = df["stars"].apply(
        lambda s: "Positive" if s >= 4 else ("Neutral" if s == 3 else "Negative")
    )

    print(f"Loaded {len(df)} reviews from {DATA_PATH}")
    print(df["true_sentiment"].value_counts())
    return df


# 2. PROCESSING - three sentiment methods
def label_from_score(score, pos_thresh=0.05, neg_thresh=-0.05):
    if score >= pos_thresh:
        return "Positive"
    elif score <= neg_thresh:
        return "Negative"
    return "Neutral"


def run_vader(df, analyzer):
    scores = df["text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
    labels = scores.apply(label_from_score)
    return scores, labels


def run_textblob(df):
    scores = df["text"].apply(lambda t: TextBlob(t).sentiment.polarity)
    labels = scores.apply(label_from_score)
    return scores, labels


def run_bert(df):
    from transformers import pipeline

    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    def classify(text):
        # SST-2 model only outputs POSITIVE/NEGATIVE - no neutral class,
        # so we treat low-confidence scores as Neutral for fairer comparison.
        result = clf(text[:512])[0]
        label, score = result["label"], result["score"]
        if score < 0.65:
            return "Neutral", score
        return ("Positive" if label == "POSITIVE" else "Negative"), score

    results = df["text"].apply(classify)
    labels = results.apply(lambda r: r[0])
    scores = results.apply(lambda r: r[1])
    return scores, labels


def process_sentiment(df):
    print("\n" + "=" * 60)
    print("STEP 2: PROCESSING (3 sentiment methods)")
    print("=" * 60)

    analyzer = SentimentIntensityAnalyzer()

    print("Running VADER...")
    df["vader_score"], df["vader_label"] = run_vader(df, analyzer)

    print("Running TextBlob...")
    df["textblob_score"], df["textblob_label"] = run_textblob(df)

    if USE_BERT:
        try:
            print("Running BERT (distilbert-sst2)... this may take a minute")
            df["bert_score"], df["bert_label"] = run_bert(df)
        except Exception as e:
            print(f"BERT skipped (transformers/torch not available: {e})")
            df["bert_score"], df["bert_label"] = None, None
    else:
        df["bert_score"], df["bert_label"] = None, None

    return df

# EVALUATION - compare each method against ground truth
def evaluate_methods(df):
    print("\n" + "=" * 60)
    print("EVALUATION vs ground-truth star ratings")
    print("=" * 60)

    methods = ["vader_label", "textblob_label"]
    if df["bert_label"].notna().any():
        methods.append("bert_label")

    accuracy_results = {}
    for method in methods:
        acc = accuracy_score(df["true_sentiment"], df[method])
        accuracy_results[method] = acc
        print(f"\n--- {method} ---")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(df["true_sentiment"], df[method], zero_division=0))

    return accuracy_results, methods

# 3. INGESTION - normalized SQLite tables
def ingest_to_database(df):
    print("\n" + "=" * 60)
    print("STEP 3: INGESTION (SQLite)")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    reviews = df[["review_id", "text", "stars", "true_sentiment"]]
    reviews.to_sql("reviews", conn, if_exists="replace", index=False)

    score_cols = ["review_id", "vader_score", "vader_label",
                  "textblob_score", "textblob_label",
                  "bert_score", "bert_label"]
    scores = df[score_cols]
    scores.to_sql("sentiment_scores", conn, if_exists="replace", index=False)

    query = """
        SELECT r.true_sentiment, s.vader_label, COUNT(*) as count
        FROM reviews r
        JOIN sentiment_scores s ON r.review_id = s.review_id
        GROUP BY r.true_sentiment, s.vader_label
        ORDER BY count DESC
    """
    print("\nSample query - true sentiment vs VADER prediction breakdown:")
    print(pd.read_sql(query, conn))

    conn.close()
    print(f"\nData written to {DB_PATH} (tables: reviews, sentiment_scores)")

# 4. VISUALIZATION
def make_visualizations(df, accuracy_results, methods):
    print("\n" + "=" * 60)
    print("STEP 4: VISUALIZATION")
    print("=" * 60)

    colors = {"Positive": "#4CAF50", "Neutral": "#9E9E9E", "Negative": "#F44336"}

    counts = df["vader_label"].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=counts.index, values=counts.values, hole=0.5,
        marker_colors=[colors[l] for l in counts.index]
    )])
    fig.update_layout(title="Sentiment Breakdown (VADER)")
    fig.write_html(str(OUTPUT_DIR / "sentiment_breakdown.html"))
    fig.write_image(str(OUTPUT_DIR / "sentiment_breakdown.png"))

    plt.figure(figsize=(7, 5))
    names = [m.replace("_label", "").upper() for m in methods]
    vals = [accuracy_results[m] for m in methods]
    plt.bar(names, vals, color=["#2196F3", "#FF9800", "#9C27B0"][:len(names)])
    plt.ylabel("Accuracy")
    plt.title("Sentiment Method Accuracy vs Star Ratings")
    plt.ylim(0, 1)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "method_accuracy_comparison.png", dpi=150)
    plt.close()

    best_method = max(accuracy_results, key=accuracy_results.get)
    labels_order = ["Positive", "Neutral", "Negative"]
    cm = confusion_matrix(df["true_sentiment"], df[best_method], labels=labels_order)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(3), labels_order)
    plt.yticks(range(3), labels_order)
    plt.xlabel("Predicted")
    plt.ylabel("True (from star rating)")
    plt.title(f"Confusion Matrix - {best_method}")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    pos_text = " ".join(df[df["vader_label"] == "Positive"]["text"])
    neg_text = " ".join(df[df["vader_label"] == "Negative"]["text"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if pos_text.strip():
        wc_pos = WordCloud(width=600, height=400, background_color="white",
                            colormap="Greens").generate(pos_text)
        axes[0].imshow(wc_pos)
    axes[0].set_title("Positive Reviews")
    axes[0].axis("off")

    if neg_text.strip():
        wc_neg = WordCloud(width=600, height=400, background_color="white",
                            colormap="Reds").generate(neg_text)
        axes[1].imshow(wc_neg)
    axes[1].set_title("Negative Reviews")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wordclouds.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["vader_score"], bins=30, color="#3F51B5", alpha=0.8)
    plt.axvline(0.05, color="green", linestyle="--", label="positive threshold")
    plt.axvline(-0.05, color="red", linestyle="--", label="negative threshold")
    plt.xlabel("VADER compound score")
    plt.ylabel("Number of reviews")
    plt.title("Distribution of Sentiment Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "score_distribution.png", dpi=150)
    plt.close()

    print(f"Charts saved to {OUTPUT_DIR}/")
    print("  - sentiment_breakdown.png / .html (interactive)")
    print("  - method_accuracy_comparison.png")
    print("  - confusion_matrix.png")
    print("  - wordclouds.png")
    print("  - score_distribution.png")

# MAIN
def main():
    df = load_data()
    df = process_sentiment(df)
    accuracy_results, methods = evaluate_methods(df)
    ingest_to_database(df)
    make_visualizations(df, accuracy_results, methods)

    # Save the enriched dataset too, useful for your report/appendix
    df.to_csv(OUTPUT_DIR / "full_results.csv", index=False)

    print("\n" + "=" * 60)
    print("DONE. Summary of findings:")
    print("=" * 60)
    for m, a in accuracy_results.items():
        print(f"  {m}: {a:.1%} accuracy vs star-rating ground truth")
    best = max(accuracy_results, key=accuracy_results.get)
    print(f"\nBest performing method: {best}")


if __name__ == "__main__":
    main()
