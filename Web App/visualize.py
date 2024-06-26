import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv("labeled_random_tweets.csv")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Extract labels and predictions
labels = df['label']
predictions = df['Prediction']

# Calculate accuracy
accuracy = accuracy_score(labels, predictions)

# Calculate precision
precision = precision_score(labels, predictions, average='weighted')

# Calculate recall
recall = recall_score(labels, predictions, average='weighted')

# Calculate F1-score
f1 = f1_score(labels, predictions, average='weighted')


# Group by label and prediction, and count occurrences
grouped = df.groupby(['label', 'Prediction']).size().unstack(fill_value=0)

# Plot the grouped data
grouped.plot(kind='bar', stacked=True)
plt.xlabel(f'Label\nAcc:{round(accuracy,5)} Pres:{round(precision,5)} Recall:{round(recall,5)} F1-score:{round(f1,5)}')
plt.ylabel('Count')
plt.title('Labels vs Predictions')
plt.legend(title='Prediction')
plt.show()
