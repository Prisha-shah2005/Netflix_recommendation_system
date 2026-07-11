import os
import getpass
from huggingface_hub import HfApi, create_repo

def deploy():
    print("--- Hugging Face Spaces Deployment Script ---")
    
    username = os.environ.get("HF_USERNAME") or input("Enter your Hugging Face Username: ").strip()
    space_name = os.environ.get("HF_SPACE_NAME") or input("Enter your Hugging Face Space Name: ").strip()
    token = os.environ.get("HF_TOKEN") or getpass.getpass("Enter your Hugging Face Write Token: ").strip()

    if not username or not space_name or not token:
        print("Error: Username, Space Name, and Write Token are required.")
        return

    repo_id = f"{username}/{space_name}"
    
    # Initialize HfApi
    api = HfApi(token=token)
    
    # Try to create the Space if it doesn't exist
    try:
        print(f"Creating Space '{repo_id}' (SDK: docker)...")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=token,
            exist_ok=True
        )
        print("Space is ready.")
    except Exception as e:
        print(f"Could not verify/create Space. It might already exist or you don't have permission. Details: {e}")
        print("Attempting to proceed with upload...")

    # Upload folder contents
    print(f"Uploading files to Space: {repo_id}...")
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[
                ".git/*",
                ".git",
                "__pycache__/*",
                "*.pyc",
                ".ipynb_checkpoints/*",
                "deploy_hf.py",
                "data/raw/*",  # ignore raw data to save space if needed
                "notebooks/*",  # ignore Jupyter notebooks
            ]
        )
        print("\nDeployment completed successfully!")
        print(f"Your application is building at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"\nUpload failed. Details: {e}")

if __name__ == "__main__":
    deploy()
