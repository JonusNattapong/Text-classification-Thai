from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Layer
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load dataset from Hugging Face
dataset = load_dataset('ZombitX64/Wisesight-Sentiment-Thai', split='train')
# Reduce dataset size for faster training
dataset = dataset.select(range(610000))  # Use only first 50k samples
texts = dataset['text']
labels = dataset['sentiment']

# Check label distribution
print(f"Total samples: {len(labels)}")
print(f"Sample labels: {labels[:10]}")  # Check first 10 labels
print(f"Unique labels: {set(labels[:100])}")  # Check unique labels in first 100 samples
print(f"Label type: {type(labels[0])}")  # Check data type

# Load tokenizer from HuggingFace Hub
tokenizer_hf = AutoTokenizer.from_pretrained("ZombitX64/Thaitokenizer")

# Tokenize Thai text using HuggingFace tokenizer
tokens = tokenizer_hf(texts, padding=False, truncation=True, max_length=512, add_special_tokens=False)['input_ids']
X = pad_sequences(tokens, maxlen=100)

# Encode labels for single-label classification
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
y = to_categorical(labels_encoded)

# Check label distribution after encoding
print(f"Label shape: {y.shape}")
print(f"Number of unique labels: {y.shape[1]}")
print(f"Label distribution (sum per class): {y.sum(axis=0)}")
print(f"Label classes: {le.classes_}")  # Show actual label names


# Simple Attention Layer
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

model = Sequential([
    Embedding(input_dim=tokenizer_hf.vocab_size, output_dim=128),
    Bidirectional(LSTM(64, return_sequences=True)),
    Attention(),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Change back to categorical_crossentropy

# Train model
model.fit(X, y, epochs=3, batch_size=32, validation_split=0.2)  # Reduce epochs from 5 to 3

# Save model and tokenizer
model.save('thai_text_classification_model.keras')  # Use new Keras format

# Save tokenizer for later use
import pickle
with open('thai_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer_hf, f)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model, tokenizer, and label encoder saved successfully!")

# Test the model with sample text
def predict_sentiment(text, model, tokenizer, label_encoder, max_len=100):
    """Predict sentiment for a given text"""
    # Tokenize the text
    tokens = tokenizer([text], padding=False, truncation=True, max_length=512, add_special_tokens=False)['input_ids']
    X_test = pad_sequences(tokens, maxlen=max_len)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Get predicted class
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = label_encoder.classes_[predicted_class]
    confidence = predictions[0][predicted_class]
    
    # Show top predictions with confidence
    top_indices = np.argsort(predictions[0])[::-1][:3]  # Top 3 predictions
    top_labels = [label_encoder.classes_[i] for i in top_indices]
    top_scores = [predictions[0][i] for i in top_indices]
    
    return predicted_label, confidence, top_labels, top_scores

# Test with sample Thai texts
test_texts = [
    "ร้านนี้อร่อยมาก ชอบมากเลย",  # Should be positive
    "แย่มาก บริการแย่ อาหารไม่อร่อย",  # Should be negative  
    "ปกติ ไม่ดีไม่แย่",  # Should be neutral
    "ดีใจมาก ประทับใจ",  # Should be positive
    "เศร้า ผิดหวัง"  # Should be negative
]

print("\n=== Testing Model ===")
for i, text in enumerate(test_texts):
    predicted_label, confidence, top_labels, top_scores = predict_sentiment(text, model, tokenizer_hf, le)
    print(f"\nTest {i+1}: {text}")
    print(f"Predicted sentiment: {predicted_label} (confidence: {confidence:.4f})")
    print(f"Top 3 predictions:")
    for label, score in zip(top_labels, top_scores):
        print(f"  {label}: {score:.4f}")