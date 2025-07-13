# Sentiment-Analysis-on-Deepseek-Reviews

🔍 Objective:
To classify user reviews of the DeepSeek app as positive or negative using deep learning and transformer-based NLP models, and to optimize model performance through hyperparameter tuning and evaluation.

Dataset Overview:
Source: Deepseek Reviews Analysis.csv

Size: 15,124 rows × 5 columns

Key Columns:
content: The textual review content.
score: User rating (1 to 5).
thumbsUpCount: Number of likes/upvotes on review.
at: Date and time of the review.

Preprocessing:
1. Dropped unnecessary columns: Unnamed: 0, thumbsUpCount, at.
2. Mapped ratings:
- Scores 3, 4, 5 → Positive (1)
- Scores 1, 2 → Negative (0)
3. Text Cleaning Steps:
- Removed HTML, emojis, punctuation.
- Lowercased text.
- Tokenization and lemmatization using NLTK.
- Stopword removal.

Exploratory Data Analysis (EDA):
- Visualized class distribution before and after mapping.
- Found class imbalance (more positive reviews).
- Used pie charts and bar plots to understand distribution.

🧠 Models Used:
1. Recurrent Neural Network (RNN) Models:
Used Keras/TensorFlow for deep learning-based text classification:

a. Simple RNN
Embedding → SimpleRNN → Dense
Result: Overfitting issues, poor performance on validation.

b. LSTM (Long Short-Term Memory)
Embedding → Bidirectional LSTM → Dense
Better generalization compared to simple RNN.

c. GRU (Gated Recurrent Unit)
Similar to LSTM but computationally lighter.
Comparable accuracy, faster training.

Training enhancements:
EarlyStopping
Class weights to manage imbalance
Tokenizer + Padding

2. Transformer-Based Model: DistilBERT
Used Hugging Face Transformers:
Tokenizer: DistilBertTokenizerFast
Model: DistilBertForSequenceClassification

Created datasets.Dataset object for HuggingFace pipeline.
Fine-tuned with Trainer API using:
Learning rate scheduling
Warm-up steps
Evaluation metrics: accuracy, precision, recall, f1-score

🧪 Results:
Model	Accuracy	Notes
Simple RNN	~74%	Overfitting on training data
LSTM	~85%	Best among RNNs
GRU	~84%	Slightly faster than LSTM
DistilBERT	~92%	Best overall, balanced precision/recal

DistilBERT also handled contextual understanding better.
Confusion matrix and classification report showed higher precision/recall in DistilBERT compared to RNNs

💡 Key Takeaways:
Text preprocessing is critical for RNN-based models.
Class imbalance affects model fairness — handled via class weights.
Transformer-based models significantly outperform traditional RNNs in text classification.
HuggingFace makes transformer fine-tuning scalable and effective.

🧾 Closing Statement:
"This project allowed me to explore a complete NLP pipeline — from raw data cleaning, EDA, and feature engineering to building and evaluating deep learning and transformer-based models. I compared traditional RNNs with a transformer (DistilBERT), and achieved the best results with the latter (~92% accuracy). This experience improved my understanding of practical NLP challenges such as data imbalance and optimization using HuggingFace's ecosystem."

