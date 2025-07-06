from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
import pickle

# ดาวน์โหลดไฟล์
model_path = hf_hub_download(repo_id="ZombitX64/Thai-text-classification", filename="thai_text_classification_model.keras")
tokenizer_path = hf_hub_download(repo_id="ZombitX64/Thai-text-classification", filename="thai_tokenizer.pkl")
label_encoder_path = hf_hub_download(repo_id="ZombitX64/Thai-text-classification", filename="label_encoder.pkl")

# โหลดโมเดลและออปเจกต์
model = load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# ทำนาย
def predict_sentiment(text):
    tokens = tokenizer([text], padding=False, truncation=True, max_length=512, add_special_tokens=False)['input_ids']
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    X_test = pad_sequences(tokens, maxlen=100)
    pred = model.predict(X_test)
    pred_class = label_encoder.classes_[pred.argmax()]
    return pred_class

print(predict_sentiment("ร้านนี้อร่อยมาก ชอบมากเลย"))
