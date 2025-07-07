
# 🧠 Named Entity Recognition (NER) with SpaCy and Transformers

This project tackles **Named Entity Recognition (NER)** using a labeled dataset and explores two approaches:

1. Custom training with **SpaCy**
2. Token classification using **HuggingFace's Transformers (BERT)**

NER is a subtask of NLP that involves identifying and classifying named entities in text (like persons, organizations, locations, etc.).

---

## 📁 Dataset Overview

The dataset used is a custom-formatted file `NER_Dataset.csv`, containing token-level annotations:

| Column        | Description                                |
|---------------|--------------------------------------------|
| `Sentence_ID` | ID of the sentence                         |
| `Word`        | Tokenized word in the sentence             |
| `POS`         | Part of Speech tag (optional)              |
| `Tag`         | BIO-formatted NER tag (e.g., B-geo, I-per) |

Example snippet:

```

| Sentence\_ID | Word          | POS | Tag |
| ------------ | ------------- | --- | --- |
| Sentence: 1  | Thousands     | NNS | O   |
| Sentence: 1  | of            | IN  | O   |
| Sentence: 1  | demonstrators | NNS | O   |
| ...          |               |     |     |

````

---

## 🧪 Workflow Summary

### 1. 📦 Data Preparation

- Loaded CSV and grouped tokens by sentence
- Converted to plain sentences and corresponding tag lists
- Preprocessed tags into BIO format

### 2. 📊 Entity Distribution

- Plotted distribution of entity labels using Seaborn

### 3. ⚙️ SpaCy NER Model Training

- Created a blank SpaCy pipeline
- Added `ner` component with dynamic label registration
- Formatted data to SpaCy's training format (start, end, label)
- Trained model over multiple epochs using minibatching
- Logged training loss per iteration

### 4. 🧪 Model Evaluation

- Extracted predicted entities from the trained model
- Aligned true vs predicted labels
- Evaluated using `sklearn`'s `classification_report`

### 5. 🧠 Transformers (Optional Extension)

- Used `transformers` pipeline with `BertForTokenClassification` for comparison

---

## 🧠 Key Results

- The SpaCy-trained NER model reached ~80% accuracy on token-level predictions.
- Class-level performance varied significantly across tags.
- Some classes (like `'a'`, `'n'`, `'v'`) were underrepresented and thus poorly predicted — highlighting **class imbalance**.

---

## 📊 Sample Output

```text
              precision    recall  f1-score   support

           O       0.84      0.84      0.84    177590
         B-geo     0.89      0.89      0.89    401168
         I-per     0.97      0.72      0.82      1811
         B-org     0.41      0.43      0.42      9978
         ...       ...       ...       ...       ...

    accuracy                           0.80    897362
   macro avg       0.47      0.44      0.45
weighted avg       0.80      0.80      0.80
````

**Note:** Labels with no predicted samples returned `UndefinedMetricWarning`. Consider applying **zero\_division=1** or rebalancing the dataset.

---

## 🔧 Tools & Libraries

* Python 🐍
* [SpaCy](https://spacy.io/)
* [Transformers by HuggingFace](https://huggingface.co/transformers/)
* scikit-learn
* Pandas, NumPy, Seaborn, Matplotlib
* PyTorch

---

## 🗂️ Project Structure

NER-Project/
│
├── NER_Dataset.csv               # Raw dataset
├── ner_training_spacy.ipynb      # Notebook for SpaCy training
├── ner_evaluation.ipynb          # Evaluation metrics
├── README.md                     # You’re here!

---

## 🔮 Possible Extensions

* Fine-tune BERT-based models (`bert-base-cased`, `roberta-base`) with `transformers`
* Use CRF layer on top of embeddings
* Visualize predictions with displaCy
* Apply on real-world articles or news data

---

## 👨‍💻 Author

**Daksh Parekh**
📍 NYU | MS in Data Science
📧 [dp3980@nyu.edu](mailto:dp3980@nyu.edu)
[🔗 LinkedIn](https://linkedin.com)
[📁 GitHub Projects](https://github.com)

---

## 📌 Notes

* SpaCy's default NER pipeline is strong but depends heavily on training data quality.
* The imbalance in tag distribution limits performance on rare entities.
* Great for demonstrating practical understanding of NER systems in **real-world pipelines**.

---

> *"The best way to understand language is to teach a machine to recognize it."*






