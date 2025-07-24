# 📩 Naive Bayes SMS Spam Classifier 

This is a machine learning project that uses the Naive Bayes algorithm to classify SMS messages as **spam** or **ham (not spam)**. The model is trained on a publicly available dataset and demonstrates key steps in text classification, including preprocessing, vectorization, training, evaluation, and model saving.
This machine learning project classifies SMS messages as **spam** or **ham (not spam)** using **Naive Bayes**, with modern ML practices like **TF-IDF vectorization** and **K-Fold cross-validation**. 

---

## 🗂️ Dataset

- Source: [SMS Spam Collection Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format: `spam.csv`
- Columns:
  - `v1` → Label (`ham` or `spam`)
  - `v2` → Message text content

---

## 📊 Project Pipeline

### 1️⃣ Import Libraries

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `sklearn`: model, metrics, preprocessing
- `nltk`: stopwords, word tokenization
- `string` : punctuation
- `re` : html tags
- `joblib`: for model saving

### 2️⃣ Data Cleaning

- Remove unnecessary columns
-  Renamed columns (`v1 → label`, `v2 → message`)
- Removed duplicates and nulls

### 3️⃣ Exploratory Data Analysis (EDA)

- Countplot of spam vs ham
- **Pairplot** to visualize numerical features (e.g., number_word,number_sentences vs label)
- Word distribution (WordCloud)

### 4️⃣ Feature Engineering
- Encoded labels: `ham → 0`, `spam → 1`
- Text preprocessing: lowercase, punctuation removal,html tag removal, stopwords
- Tokenizing and lemmatizing words
- TF-IDF vectorization using `TfidfVectorizer`
### Train Test split

- use 80% for train and 20% for test the model

### 5️⃣ Model Building – Naive Bayes

- **MultinomialNB** classifier
- Used **K-Fold Cross Validation** (`StratifiedKFold`)

### 6️⃣Evaluation

- accurcy score,precision and recall score
- Mean accuracy across K-folds
- Confusion matrix
- Classification report

---
### 7 Saved the Model

- Saved model in `NB_sms_classifier.joblib`

  ---

## 📈 Visuals

- ✅ Pairplot with `seaborn.pairplot()`
- ✅ Confusion matrix heatmap
  

---

## 🧠 Model & Techniques Used

| Technique | Description |
|----------|-------------|
| TF-IDF Vectorizer | Transforms text to weighted word features |
| MultinomialNB | Fast probabilistic classifier for text |
| K-Fold CV | More reliable accuracy estimation |
| Pairplot | Visualizes relation between multiple numeric features |

---

## 🛠️ Setup Instructions

1. Clone the repository
2. Download the dataset from Kaggle and place it in the project folder
3. Install required dependencies using `requirements.txt`
4. Run the notebook or script to train the model and generate plots

## 📦 Sample Requirements (`requirements.txt`)
