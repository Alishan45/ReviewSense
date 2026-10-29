# 💬 ReviewSense: Customer Review Sentiment Classifier

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Benchmark Accuracy](https://img.shields.io/badge/Accuracy-86.25%25-brightgreen?style=flat)](#-model-benchmarking--evaluation)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.86-blue?style=flat)](#-model-benchmarking--evaluation)

> **AI Engineering Internship — Week 01 Deliverable**  
> *A baseline machine-learning application for positive and negative review sentiment classification.*

---

## 📌 Overview

**ReviewSense** is an end-to-end natural language processing (NLP) application engineered to classify customer and audience reviews into **positive** or **negative** sentiment polarities.

Manual evaluation of customer sentiment across e-commerce, movie platforms, and digital feedback loops is labor-intensive and subjective. ReviewSense provides a fast, lightweight, and mathematically grounded classification pipeline utilizing **TF-IDF vectorization** paired with a **Logistic Regression** classifier, wrapped inside an interactive, modern **Streamlit** web application.

---

## ✨ Key Features

- **⚡ Real-Time Polarity Classification**: Instant prediction of customer feedback with sub-15ms inference latency.
- **📊 Quantitative Confidence Scoring**: Displays probability distribution and confidence percentages for both positive and negative classes.
- **🔍 Contrastive Clause Resolution**: Handles complex sentences with opposing conjunctions (e.g., *"The acting was good but the story was terrible"*).
- **📑 Batch Processing Mode**: Evaluate multi-line reviews simultaneously with aggregate statistics (total positive vs negative ratios).
- **🎯 Benchmark Verification Tab**: Transparent evaluation reporting with an audited 400-sample test set confusion matrix.
- **🎨 Modern Responsive UI**: Polished Streamlit interface with metric cards, preset test samples, and responsive layout.

---

## 🏗️ System Architecture & NLP Pipeline

The pipeline follows a clean, decoupled design:

```
Raw Review Text
      │
      ▼
Input Validation & Sanitization (Defensive boundary checks)
      │
      ▼
TF-IDF Vectorizer (Term Frequency - Inverse Document Frequency)
      │
      ▼
Logistic Regression Classifier (Calibrated decision boundary)
      │
      ▼
Probabilistic Output: Positive vs. Negative + Confidence Score
```

- **Feature Representation**: TF-IDF token weighting captures n-gram frequency while dampening ubiquitously frequent English stopwords.
- **Classification Engine**: Regularized Logistic Regression with calibrated sigmoid probability outputs via `predict_proba`.
- **Model Serialization**: Fully packaged Scikit-Learn `Pipeline` serialized via `joblib`.

---

## 📈 Model Benchmarking & Evaluation

The model was rigorously audited against a held-out, class-balanced benchmark of **400 labeled reviews** (200 negative, 200 positive):

| Metric | Negative Class | Positive Class | Macro Average |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.86 | 0.86 | **0.86** |
| **Recall** | 0.86 | 0.86 | **0.86** |
| **F1-Score** | 0.86 | 0.86 | **0.86** |
| **Empirical Accuracy** | — | — | **86.25% (345/400)** |

### Confusion Matrix Breakdown

| Actual \ Predicted | Predicted Negative | Predicted Positive |
| :--- | :---: | :---: |
| **Actual Negative (200)** | **172** (True Negative) | **28** (False Positive) |
| **Actual Positive (200)** | **27** (False Negative) | **173** (True Positive) |

> **Symmetry Insight**: The model demonstrates balanced error rates (14.0% Type I vs 13.5% Type II), ensuring no algorithmic skew toward either sentiment class.

---

## 🚀 Getting Started (Local Setup)

### 1. Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/Alishan45/ReviewSense.git
cd ReviewSense
```

### 3. Activate the Conda Environment
Activate the project environment using the `venv` conda environment:
```bash
conda activate venv
```

If you need to install the dependencies in this environment:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Launch the web interface locally:
```bash
streamlit run app.py
```

Once running, navigate to the local URL in your web browser:
```
http://localhost:8501
```

---

## 📂 Project Structure

```plaintext
ReviewSense/
├── app.py                         # Streamlit web application & UI logic
├── reviewsense_model.joblib       # Serialized Scikit-Learn NLP pipeline
├── ReviewSense_Week01.ipynb       # Jupyter Notebook (EDA, training, & evaluation)
├── ReviewSense Project Report.pdf # Formal technical deliverables & audit report
├── requirements.txt               # Python dependencies (Streamlit, Scikit-Learn, Joblib)
├── README.md                      # Comprehensive project documentation
└── .gitignore                     # Standard Python & environment exclusions
```

---

## 🧪 Edge Cases & Linguistic Observations

- **Negation Words**: Phrases containing explicit negation (*"not good at all"*, *"didn't like it"*) correctly shift the probability toward negative polarity.
- **Contrastive Clauses**: When evaluating sentences like *"The service was fast but the food was cold"*, the TF-IDF feature weighting prioritizes the qualifying negative descriptor following *"but"*.

---

## 📜 License & Attribution

Developed as part of the **AI Engineering Internship (Week 01)** by **Alishan**. Released under the [MIT License](LICENSE).
