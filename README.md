
# Sentiment Analysis

The task was to build a sentiment analysis solution, explain how it works, and evaluate how well it performs. Basically: can a program tell if a review is positive, negative, or neutral, and how good is it at that.

We did everything locally in Python instead of using Azure, since the assignment says Azure isn't mandatory. Data goes into a SQLite database instead of Azure SQL, and the charts are made with matplotlib/plotly.

Link to dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## What it does

1. Loads a CSV of Amazon reviews (text + star rating)
2. Runs the text through three different sentiment methods: VADER, TextBlob, and a BERT model
3. Checks how accurate each one is by comparing it to the star rating (treated as the "true" answer)
4. Saves everything into a local database
5. Makes a few charts so you can actually see the results

## Setup

```
python3 -m venv sentiment_env
source sentiment_env/bin/activate
pip install -r requirements.txt
python -m textblob.download_corpora
```

## Running it

To test with fake sample data first (so you know it works before using real data):

```
python scripts/sentiment_analysis.py
```

For the actual dataset, we used the Amazon Fine Food Reviews dataset from Kaggle. It's pretty big (500k+ rows) so I only used a sample of 40,000 - running BERT on the full thing would take forever on a regular CPU.


## Results

Everything gets saved into the `outputs/` folder:
- `sentiment_analysis.db` — the SQLite database with two tables (reviews, sentiment_scores)
- a handful of charts (accuracy comparison, confusion matrix, word clouds, sentiment breakdown, score distribution)
- `full_results.csv` — the full dataset with all the sentiment scores attached, in case you want to dig into it more

On the run with 40,000 reviews, VADER actually came out on top with about 80% accuracy, TextBlob and BERT were both around 76-77%. The interesting part wasn't really which one "won" though all three completely struggled with neutral reviews. Makes sense honestly, a 3 star review is often just someone being lukewarm or having mixed feelings, and none of these methods really "get" that.

## Folder structure

```
sentiment-project/
├── requirements.txt
├── README.md
├── data/
│   └── Reviews.csv
├── scripts/
│   └── sentiment_analysis.py
└── outputs/
```

