import os
import gdown
import zipfile

# Google Drive File IDs
DATA_ZIP_ID = "19Bn7_zEFxH7kEYQB_TtrQ1ExFFPOluyX"
MODEL_ZIP_ID = "1Qp04bRjPfOwUFSzBww4H2oWrVUiVw0SI"


def download_and_extract(google_id, output_zip, target_dir):
    url = f"https://drive.google.com/uc?id={google_id}"

    print(f"\n===== Downloading: {output_zip} =====")
    gdown.download(url, output_zip, quiet=False)

    print(f"Extracting to: {target_dir}")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(output_zip)
    print(f"✓ {output_zip} completed.\n")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    download_and_extract(DATA_ZIP_ID, "data.zip", "data")
    download_and_extract(MODEL_ZIP_ID, "models.zip", "models")

    print("===== All assets downloaded and extracted successfully! =====")
