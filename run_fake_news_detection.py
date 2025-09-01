#!/usr/bin/env python3
"""
Fake News Detection Project - Complete Implementation
This script runs the entire fake news detection pipeline and saves all visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from wordcloud import WordCloud
from collections import Counter
import os

# For reproducibility
np.random.seed(42)

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Set matplotlib backend to save figures
plt.switch_backend('Agg')

print("Step 1: Loading and preparing data...")

# 1. Load the datasets
fake_df = pd.read_csv('archive (2)/Fake.csv')
true_df = pd.read_csv('archive (2)/True.csv')

# Add a label column: 0 for Fake, 1 for True
fake_df['label'] = 0
true_df['label'] = 1

# Combine into a single DataFrame
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset loaded: {len(df)} total articles")
print(f"Fake news: {len(df[df['label']==0])}")
print(f"Real news: {len(df[df['label']==1])}")

# 2. Handle Missing Values
print("\nStep 2: Data cleaning and preprocessing...")

# Check missing values
missing = df.isnull().sum()
print("Missing values per column:")
print(missing)

# Fill missing values in 'title' and 'text' with empty string
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')

# Combine Title and Text into a Single Feature
df['content'] = df['title'] + ' ' + df['text']

# 3. Text Cleaning
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize and remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

print("Cleaning text (this may take a few minutes)...")
df['clean_content'] = df['content'].apply(clean_text)

# 4. Exploratory Data Analysis
print("\nStep 3: Exploratory Data Analysis...")

# Class balance visualization
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df)
plt.title('Class Distribution (0=Fake, 1=True)', fontsize=14)
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig('visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("Class distribution saved to visualizations/class_distribution.png")

# Word frequency analysis
def get_top_n_words(corpus, n=None):
    words = ' '.join(corpus).split()
    counter = Counter(words)
    return counter.most_common(n)

# Top 20 words in fake news
fake_words = get_top_n_words(df[df['label']==0]['clean_content'], 20)
print('\nTop 20 words in Fake News:')
for word, count in fake_words:
    print(f'{word}: {count}')

# Top 20 words in real news
real_words = get_top_n_words(df[df['label']==1]['clean_content'], 20)
print('\nTop 20 words in Real News:')
for word, count in real_words:
    print(f'{word}: {count}')

# Word clouds
plt.figure(figsize=(15, 6))

# Fake news word cloud
wordcloud_fake = WordCloud(width=600, height=400, background_color='white').generate(' '.join(df[df['label']==0]['clean_content']))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title('Fake News Word Cloud', fontsize=14)

# Real news word cloud
wordcloud_real = WordCloud(width=600, height=400, background_color='white').generate(' '.join(df[df['label']==1]['clean_content']))
plt.subplot(1, 2, 2)
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title('Real News Word Cloud', fontsize=14)

plt.tight_layout()
plt.savefig('visualizations/word_clouds.png', dpi=300, bbox_inches='tight')
plt.close()

print("Word clouds saved to visualizations/word_clouds.png")

# 5. Feature Engineering
print("\nStep 4: Feature Engineering...")

from sklearn.feature_extraction.text import TfidfVectorizer

# Use only the cleaned content with reduced features to avoid memory issues
tfidf = TfidfVectorizer(max_features=2000)  # Reduced from 5000 to 2000
X = tfidf.fit_transform(df['clean_content'])  # Keep as sparse matrix
y = df['label'].values

print(f'TF-IDF matrix shape: {X.shape}')

# 6. Model Building
print("\nStep 5: Model Building...")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f'Train shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')

# Train models (using sparse matrices)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),  # Reduced n_estimators
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=50)  # Reduced n_estimators
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
    predictions[name] = y_pred
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} ROC-AUC: {roc_auc:.4f}")

# 7. Model Evaluation
print("\nStep 6: Model Evaluation...")

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name} Confusion Matrix')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print("Confusion matrices saved to visualizations/confusion_matrices.png")

# Model comparison
plt.figure(figsize=(12, 5))

# Accuracy comparison
plt.subplot(1, 2, 1)
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)

# ROC-AUC comparison
plt.subplot(1, 2, 2)
roc_aucs = [results[name]['roc_auc'] for name in model_names]
plt.bar(model_names, roc_aucs, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
plt.title('Model ROC-AUC Comparison')
plt.ylabel('ROC-AUC')
plt.xticks(rotation=45)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Model comparison saved to visualizations/model_comparison.png")

# 8. Hyperparameter Tuning
print("\nStep 7: Hyperparameter Tuning...")

from sklearn.model_selection import GridSearchCV
import joblib

# Tune Logistic Regression (simplified grid search)
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear']
}

gs_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), 
                    param_grid, cv=2, scoring='f1', n_jobs=1)  # Reduced cv and n_jobs
gs_lr.fit(X_train, y_train)

print(f'Best Logistic Regression params: {gs_lr.best_params_}')
print(f'Best Logistic Regression F1 score: {gs_lr.best_score_:.4f}')

y_pred_gs = gs_lr.predict(X_test)
print(f'Tuned Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_gs):.4f}')

# 9. Save Final Model
print("\nStep 8: Saving Final Model...")

# Save the tuned logistic regression model
joblib.dump(gs_lr.best_estimator_, 'best_fake_news_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Models saved as 'best_fake_news_model.pkl' and 'tfidf_vectorizer.pkl'")

# 10. Feature Importance
print("\nStep 9: Feature Importance Analysis...")

feature_names = tfidf.get_feature_names_out()
coefs = gs_lr.best_estimator_.coef_[0]

# Top 20 positive and negative features
sorted_idx = np.argsort(coefs)

print('Top 20 words for REAL news:')
for idx in sorted_idx[-20:][::-1]:
    print(f'{feature_names[idx]}: {coefs[idx]:.4f}')

print('\nTop 20 words for FAKE news:')
for idx in sorted_idx[:20]:
    print(f'{feature_names[idx]}: {coefs[idx]:.4f}')

# Feature importance visualization
plt.figure(figsize=(12, 8))

# Top 10 words for real news
plt.subplot(2, 1, 1)
top_real_idx = sorted_idx[-10:][::-1]
plt.barh(range(10), [coefs[idx] for idx in top_real_idx], color='green', alpha=0.7)
plt.yticks(range(10), [feature_names[idx] for idx in top_real_idx])
plt.title('Top 10 Words Indicating REAL News')
plt.xlabel('Coefficient Value')

# Top 10 words for fake news
plt.subplot(2, 1, 2)
top_fake_idx = sorted_idx[:10]
plt.barh(range(10), [coefs[idx] for idx in top_fake_idx], color='red', alpha=0.7)
plt.yticks(range(10), [feature_names[idx] for idx in top_fake_idx])
plt.title('Top 10 Words Indicating FAKE News')
plt.xlabel('Coefficient Value')

plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Feature importance saved to visualizations/feature_importance.png")

# 11. Final Results Summary
print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)

print("\nModel Performance:")
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest performing model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")

print(f"\nVisualizations saved in 'visualizations/' folder:")
print("- class_distribution.png")
print("- word_clouds.png") 
print("- confusion_matrices.png")
print("- model_comparison.png")
print("- feature_importance.png")

print(f"\nModels saved:")
print("- best_fake_news_model.pkl")
print("- tfidf_vectorizer.pkl")

print("\nProject completed successfully!") 