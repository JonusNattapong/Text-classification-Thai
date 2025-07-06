from huggingface_hub import HfApi, HfFolder, upload_file

# Set your Hugging Face token (replace 'your_token_here' with your actual token)
# You can get your token from https://huggingface.co/settings/tokens
HF_TOKEN = 'your_token_here'
REPO_ID = 'your-username/thai-text-classification-model'  # Change to your repo name

api = HfApi()
HfFolder.save_token(HF_TOKEN)

# Upload model
api.upload_file(
    path_or_fileobj='thai_text_classification_model.keras',
    path_in_repo='thai_text_classification_model.keras',
    repo_id=REPO_ID,
    token=HF_TOKEN
)

# Upload tokenizer
api.upload_file(
    path_or_fileobj='thai_tokenizer.pkl',
    path_in_repo='thai_tokenizer.pkl',
    repo_id=REPO_ID,
    token=HF_TOKEN
)

# Upload label encoder
api.upload_file(
    path_or_fileobj='label_encoder.pkl',
    path_in_repo='label_encoder.pkl',
    repo_id=REPO_ID,
    token=HF_TOKEN
)

print('Upload complete!')
