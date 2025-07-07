
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder



print("[INFO] Loading dataset from Hugging Face ...")
dataset = load_dataset('ZombitX64/Wisesight-Sentiment-Thai', split='train')
print("[INFO] Dataset loaded. Selecting subset ...")
dataset = dataset.select(range(600000))  # Use 50k samples for demo
texts = dataset['text']
labels = dataset['sentiment']
print("[INFO] Dataset ready.")


print("[INFO] Encoding labels ...")
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_labels = len(le.classes_)
print(f"Total samples: {len(labels)}")
print(f"Label classes: {le.classes_}")


MODEL_NAME = "xlm-roberta-base"  # Multilingual XLM-RoBERTa (supports Thai)
print(f"[INFO] Loading tokenizer and model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("[INFO] Tokenizing texts ...")
MAX_LENGTH = 128
encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH)
print("[INFO] Tokenization complete.")

# Convert to tf.data.Dataset
def gen():
    for i in range(len(texts)):
        yield ({'input_ids': encodings['input_ids'][i],
                'attention_mask': encodings['attention_mask'][i]}, labels_encoded[i])


print("[INFO] Preparing tf.data.Dataset ...")
train_dataset = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        {
            'input_ids': tf.TensorSpec(shape=(MAX_LENGTH,), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(MAX_LENGTH,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
)
train_dataset = train_dataset.shuffle(10000).batch(16).prefetch(tf.data.AUTOTUNE)
print("[INFO] Dataset ready for training.")




print("[INFO] Loading pre-trained model ...")
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print("[INFO] Model loaded and compiled.")

print("[INFO] Training ...")
model.fit(train_dataset, epochs=2)  # You can increase epochs if you have GPU
print("[INFO] Training complete.")


print("[INFO] Saving model and tokenizer ...")
model.save_pretrained('thai_text_transformer_model')
tokenizer.save_pretrained('thai_text_transformer_model')

# Save label encoder
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("[INFO] Transformer model, tokenizer, and label encoder saved successfully!")


# Test the model with sample text
def predict_sentiment(text, model, tokenizer, label_encoder, max_len=128):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_len, return_tensors='tf')
    logits = model(inputs)[0]
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred_idx = np.argmax(probs)
    pred_label = label_encoder.classes_[pred_idx]
    confidence = probs[pred_idx]
    top_indices = np.argsort(probs)[::-1][:3]
    top_labels = [label_encoder.classes_[i] for i in top_indices]
    top_scores = [probs[i] for i in top_indices]
    return pred_label, confidence, top_labels, top_scores

test_texts = [
    "ร้านนี้อร่อยมาก ชอบมากเลย",  # Should be positive
    "แย่มาก บริการแย่ อาหารไม่อร่อย",  # Should be negative
    "ปกติ ไม่ดีไม่แย่",  # Should be neutral
    "ดีใจมาก ประทับใจ",  # Should be positive
    "เศร้า ผิดหวัง"  # Should be negative
]

print("\n=== Testing Transformer Model ===")
for i, text in enumerate(test_texts):
    predicted_label, confidence, top_labels, top_scores = predict_sentiment(text, model, tokenizer, le)
    print(f"\nTest {i+1}: {text}")
    print(f"Predicted sentiment: {predicted_label} (confidence: {confidence:.4f})")
    print(f"Top 3 predictions:")
    for label, score in zip(top_labels, top_scores):
        print(f"  {label}: {score:.4f}")