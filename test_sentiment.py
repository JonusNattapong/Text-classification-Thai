import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model, tokenizer, and label encoder
model = load_model('thai_text_classification_model.keras')
with open('thai_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def predict_sentiment(text, model, tokenizer, label_encoder, max_len=100):
    tokens = tokenizer([text], padding=False, truncation=True, max_length=512, add_special_tokens=False)['input_ids']
    X_test = pad_sequences(tokens, maxlen=max_len)
    predictions = model.predict(X_test)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = label_encoder.classes_[predicted_class]
    confidence = predictions[0][predicted_class]
    top_indices = np.argsort(predictions[0])[::-1][:3]
    top_labels = [label_encoder.classes_[i] for i in top_indices]
    top_scores = [predictions[0][i] for i in top_indices]
    return predicted_label, confidence, top_labels, top_scores

# Test with sample Thai texts
test_texts = [
    "ร้านนี้อร่อยมาก ชอบมากเลย",  # Should be positive
    "แย่มาก บริการแย่ อาหารไม่อร่อย",  # Should be negative
    "ปกติ ไม่ดีไม่แย่",  # Should be neutral
    "ดีใจมาก ประทับใจ",  # Should be positive
    "เศร้า ผิดหวัง",  # Should be negative
    # ยากระดับ 10
    "อาหารอร่อยแต่บริการช้ามากจนรู้สึกหงุดหงิด แม้จะชอบบรรยากาศร้านแต่คงไม่กลับมาอีก",  # ควรเป็น negative หรือ mixed
    "พนักงานพูดจาดีแต่ลืมออเดอร์บ่อย อาหารก็มาไม่ครบแต่รสชาติดีมาก",  # ควรเป็น neutral หรือ mixed
    "ตอนแรกคิดว่าจะไม่อร่อยแต่พอลองแล้วกลับชอบมากกว่าที่คิด",  # ควรเป็น positive
    "ผิดหวังกับคุณภาพอาหารรอบนี้ ทั้งที่เคยมาก่อนแล้วประทับใจมาก",  # ควรเป็น negative
    "ร้านนี้เคยดีแต่หลังๆคุณภาพตกมาก เสียดายที่เคยแนะนำเพื่อนมา",  # ควรเป็น negative
    "แม้จะต้องรอนานแต่พอได้กินแล้วก็รู้สึกว่าคุ้มค่า รสชาติยังดีเหมือนเดิม",  # ควรเป็น positive หรือ neutral
    "รีวิวนี้ขอให้คะแนนกลางๆ เพราะอาหารบางอย่างดีแต่บางอย่างแย่ บริการก็ไม่แน่นอน",  # ควรเป็น neutral
]

print("\n=== Testing Model ===")
for i, text in enumerate(test_texts):
    predicted_label, confidence, top_labels, top_scores = predict_sentiment(text, model, tokenizer, label_encoder)
    print(f"\nTest {i+1}: {text}")
    print(f"Predicted sentiment: {predicted_label} (confidence: {confidence:.4f})")
    print(f"Top 3 predictions:")
    for label, score in zip(top_labels, top_scores):
        print(f"  {label}: {score:.4f}")
