import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from datasets import load_dataset

# Load model, tokenizer, and label encoder
model = load_model('thai_text_classification_model.keras')
with open('thai_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load a validation set (use last 10,000 samples for evaluation)
dataset = load_dataset('ZombitX64/Wisesight-Sentiment-Thai', split='train')
dataset = dataset.select(range(600000, 610000))
texts = dataset['text']
labels = dataset['sentiment']

# Encode labels
labels_encoded = label_encoder.transform(labels)

# Tokenize
tokens = tokenizer(texts, padding=False, truncation=True, max_length=512, add_special_tokens=False)['input_ids']
X = pad_sequences(tokens, maxlen=100)

# Predict
y_pred_prob = model.predict(X, batch_size=128)
y_pred = np.argmax(y_pred_prob, axis=1)

# Classification report
report = classification_report(labels_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True)
print(classification_report(labels_encoded, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(labels_encoded, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

# Plot per-class accuracy
acc_per_class = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(8, 4))
plt.bar(label_encoder.classes_, acc_per_class)
plt.ylabel('Accuracy')
plt.xlabel('Class')
plt.title('Per-Class Accuracy')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('per_class_accuracy.png')
plt.close()

# Plot per-class F1-score
f1_scores = [report[label]['f1-score'] for label in label_encoder.classes_]
plt.figure(figsize=(8, 4))
plt.bar(label_encoder.classes_, f1_scores, color='skyblue')
plt.ylabel('F1-score')
plt.title('Per-class F1-score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('f1_scores.png')
plt.close()

print('Saved confusion_matrix.png, per_class_accuracy.png, and f1_scores.png')

# Plot ROC curves (one-vs-rest)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
n_classes = len(label_encoder.classes_)
y_true_bin = label_binarize(labels_encoded, classes=range(n_classes))
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label_encoder.classes_[i]} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (One-vs-Rest)')
plt.legend()
plt.show()
