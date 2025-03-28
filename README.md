Sentiment Classification using Naïve Bayes

📌 Overview
This project performs text classification using the Naïve Bayes algorithm. It takes a dataset containing text data and their corresponding sentiment categories (-1, 0, 1), processes the data using TF-IDF vectorization, trains a classifier, and evaluates its performance using various visualization techniques.

📂 Dataset
The dataset consists of two columns:
clean_text: Preprocessed text data.
category: Label representing sentiment (-1: Negative, 0: Neutral, 1: Positive).

🔧 Technologies Used

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn

WordCloud

🚀 Installation
Clone the repository:

git clone https://github.com/your-username/sentiment-classification.git
cd sentiment-classification

Install required dependencies:
pip install -r requirements.txt

🛠️ Implementation Steps
Load Dataset: Read the dataset using Pandas.
Preprocess Data: Handle missing values and text cleaning.
Feature Extraction: Convert text to numerical representation using TF-IDF Vectorization.
Model Training: Train a Multinomial Naïve Bayes Classifier.

Evaluation & Visualization:

> Confusion Matrix
> Classification Report
> Word Cloud

📊 Visualizations

1️⃣ Confusion Matrix

Displays the performance of the classification model:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

2️⃣ Classification Report (Heatmap)
from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).T

sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap='coolwarm')
plt.title('Classification Report')
plt.show()

3️⃣ Word Cloud
from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=400, background_color='black').generate(" ".join(df['clean_text']))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Frequent Words')
plt.show()

🏆 Results
The Naïve Bayes classifier successfully classifies text into three sentiment categories.

Visualization techniques provide insights into classification performance.

👨‍💻 Contribution
Feel free to fork the repository and contribute!

📜 License
This project is licensed under the MIT License.

🚀 Happy Coding!

