# Fake News Detection Project

An end-to-end machine learning project that accurately classifies whether a news article is real or fake using various NLP and ML techniques.

## 🎯 Project Objective

Build a robust Fake News Detection classifier that can accurately classify whether a news article is real or fake using the Fake News Dataset from Kaggle.

## 📊 Dataset

- **Source**: [Fake News Dataset from Kaggle](https://www.kaggle.com/clmentbisaillon/fake-news-dataset)
- **Size**: 44,898 articles (23,481 fake, 21,417 real)
- **Features**: title, text, subject, date
- **Target**: Binary classification (0 = Fake, 1 = Real)

## 🚀 Project Workflow

### 1. Setup & Data Import
- Load and combine fake and real news datasets
- Add binary labels (0 for fake, 1 for real)
- Shuffle data for unbiased training

### 2. Data Cleaning & Preprocessing
- Handle missing values
- Combine title and text into single feature
- Text cleaning: lowercasing, punctuation removal, stopword removal, lemmatization

### 3. Exploratory Data Analysis (EDA)
- Class balance analysis
- Word frequency analysis for fake vs real news
- Word cloud visualizations

### 4. Feature Engineering
- TF-IDF vectorization (2000 features)
- Sparse matrix representation for memory efficiency

### 5. Model Building
- **Baseline Models**: Logistic Regression, Naive Bayes
- **Advanced Models**: Random Forest, XGBoost
- Train/test split (80/20) with stratification

### 6. Model Evaluation
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix visualization
- Model comparison charts

### 7. Hyperparameter Tuning
- GridSearchCV for Logistic Regression optimization

### 8. Model Deployment
- Save best model and vectorizer for reuse

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **XGBoost** | **0.9983** | **0.9983** |
| Random Forest | 0.9980 | 0.9980 |
| Logistic Regression (Tuned) | 0.9939 | 0.9875 |
| Naive Bayes | 0.9352 | 0.9348 |

**Best Model**: XGBoost with 99.83% accuracy

### Key Insights

**Top words indicating REAL news:**
- reuters, said, washington, wednesday, tuesday, thursday, friday, statement, london, democratic

**Top words indicating FAKE news:**
- via, read, video, image, gop, wire, breaking, featured, watch, hillary

## 📁 Project Structure

```
Fake news Detect/
├── README.md                           # Project documentation
├── run_fake_news_detection.py          # Complete implementation script
├── FakeNewsDetection.ipynb             # Jupyter notebook with step-by-step analysis
├── visualizations.ipynb                # Visualization notebook
├── best_fake_news_model.pkl            # Trained XGBoost model
├── tfidf_vectorizer.pkl                # TF-IDF vectorizer
├── .gitignore                          # Git ignore file
├── archive (2)/                        # Dataset folder
│   ├── Fake.csv                        # Fake news dataset
│   └── True.csv                        # Real news dataset
└── visualizations/                     # Generated visualizations
    ├── class_distribution.png          # Class balance chart
    ├── word_clouds.png                 # Word clouds for fake vs real
    ├── confusion_matrices.png          # Confusion matrices for all models
    ├── model_comparison.png            # Model performance comparison
    └── feature_importance.png          # Feature importance analysis
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Processing**: pandas, numpy
- **NLP**: nltk, re
- **Visualization**: matplotlib, seaborn, wordcloud
- **Machine Learning**: scikit-learn, xgboost
- **Model Persistence**: joblib

## 📦 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "Fake news Detect"
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud xgboost joblib
   ```

3. **Download NLTK data** (automatically handled by the script)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 🚀 Usage

### Run Complete Pipeline
```bash
python run_fake_news_detection.py
```

### Use Trained Model for Predictions
```python
import joblib
import pandas as pd

# Load the trained model and vectorizer
model = joblib.load('best_fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example prediction
text = "Your news article text here..."
cleaned_text = clean_text(text)  # Use the same cleaning function
features = vectorizer.transform([cleaned_text])
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0]

print(f"Prediction: {'FAKE' if prediction == 0 else 'REAL'}")
print(f"Confidence: {max(probability):.2%}")
```

## 📊 Visualizations

The project includes comprehensive visualizations saved in the `visualizations/` folder:

1. **Class Distribution**: Shows the balance between fake and real news articles
2. **Word Clouds**: Visual representation of most frequent words in fake vs real news
3. **Confusion Matrices**: Performance breakdown for each model
4. **Model Comparison**: Bar charts comparing accuracy and ROC-AUC across models
5. **Feature Importance**: Top words that indicate real vs fake news

## 🔍 Key Findings

1. **Model Performance**: XGBoost achieved the highest accuracy (99.83%)
2. **Feature Importance**: Words like "reuters", "washington" strongly indicate real news
3. **Fake News Indicators**: Words like "video", "breaking", "gop" are common in fake news
4. **Class Balance**: Dataset is relatively balanced (52% fake, 48% real)

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the model performance
- Adding new features or visualizations
- Optimizing the code
- Adding more datasets

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Created as part of a comprehensive data science project demonstrating end-to-end machine learning workflow.

---

**Note**: This project demonstrates best practices in data science including proper data preprocessing, model evaluation, visualization, and documentation. 