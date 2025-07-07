import os
from huggingface_hub import HfApi, HfFolder, upload_file

# HF_TOKEN: load from environment to avoid committing secrets
token = os.getenv('HF_TOKEN')
if token is None:
    raise EnvironmentError('Please set the HF_TOKEN environment variable before running this script')

# Set your Hugging Face token (replace 'your_token_here' with your actual token)
# You can get your token from https://huggingface.co/settings/tokens
HF_TOKEN = token
REPO_ID = 'ZombitX64/ThaiSentiment-BiLSTM-Attn'  # Change to your repo name

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
