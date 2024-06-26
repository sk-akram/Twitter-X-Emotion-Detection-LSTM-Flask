import pandas as pd
import random

# Load the CSV file into a DataFrame
df = pd.read_csv("training.csv")

# Select 5 random rows
random_rows = df.sample(n=100)

# Save the selected rows to a new CSV file
random_rows.to_csv("random_tweets.csv", index=False)

print("Random tweets saved to random_tweets.csv")
