#Sentiment Analysis on Twitter Data

##Project Description
This project performs sentiment analysis on the Sentiment140 dataset, which contains 1.6 million tweets labeled as positive, negative, or neutral. The goal is to classify tweets as either positive or negative using various machine learning techniques.


# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [License](#license)

# Installation
To set up the project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your_username/sentiment-analysis.git
   cd sentiment-analysis
   ```
2. Create a virtual environment and activate it
```python
python -m venv venv
source venv/bin/activate
```
3. install required dependencies
```python
pip install -r requirements.txt
```

4. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

# Usage
1. load the dataset
```python
df = pd.read_csv(r"C:\Users\Lenovo\Documents\SASM\dataset\training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)
df.columns = ["target", 'ids', 'data', 'flag', 'user', 'text']
```

2. Clean and preprocess the text
```python
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['tokens'] = df['clean_text'].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])
stemmer = PorterStemmer()
df['processed_text'] = df['tokens'].apply(lambda x: " ".join(x))
```

3. Vectorize the text using TFIDF
```python
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed_text'])
y = df['target'].apply(lambda x: 1 if x == 4 else 0)
```

4. Split the data into training andf testing sets
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
# Project Structure
```bash
sentiment-analysis/
├── data/
│   └── training.1600000.processed.noemoticon.csv  # Dataset file
├── notebooks/
│   └── sentiment_analysis.ipynb                   # Jupyter notebook with the code
├── src/
│   ├── data_processing.py                         # Data processing scripts
│   ├── model_training.py                          # Model training scripts
│   └── evaluation.py                              # Model evaluation scripts
├── README.md                                      # Project README file
└── requirements.txt                               # List of dependencies
```

# Data Processing
The dataset is loaded and preprocessed using the following steps:
- Remove URLs, mentions, hashtags, numbers, and punctuation.
- Convert text to lowercase.
- Tokenize the text.
- Remove stopwords.
- Apply stemming.

# Model Training
The LightGBM model is trained with the following parameters:
```python
params = {
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.05,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'is_unbalance': True,
    'random_state': 42
}

model = lgb.train(
    params,
    lgb.Dataset(X_train, label=y_train),
    num_boost_round=20000,
    valid_sets=[lgb.Dataset(X_train, label=y_train), lgb.Dataset(X_test, label=y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
)
```

Model Evaluation
```bash
Accuracy: 0.79
              precision    recall  f1-score   support

           0       0.80      0.76      0.78    159494
           1       0.77      0.81      0.79    160506

    accuracy                           0.79    320000
   macro avg       0.79      0.79      0.78    320000
weighted avg       0.79      0.79      0.78    320000
```
# License
This README file provides a comprehensive overview of your project, including its description, installation instructions, usage, project structure, data processing, model architecture, training, evaluation, web application, and prediction. It also includes sections for contributing and licensing, which are important for open-source projects.
