from huggingface_hub import snapshot_download
import os
import urllib.request
from tqdm import tqdm
base_path="."
def download_file(url, download_file_path):
    os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
    if os.path.exists(download_file_path):
        os.remove(download_file_path)
    try:
        with urllib.request.urlopen(url) as request, open(download_file_path, 'wb') as output, \
             tqdm(total=int(request.headers.get('Content-Length', 0)), desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
            while chunk := request.read(8192):
                output.write(chunk)
                progress.update(len(chunk))
        print(f"Download successful! Saved at: {download_file_path}")
    except Exception as e:
        print(f"Error: {e}")


def download_model(repo_id, target_folder, ignore_patterns=None):
    os.makedirs(target_folder, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=target_folder, ignore_patterns=ignore_patterns or [])
    print(f"Downloaded {repo_id} to {target_folder}")


model_folder = f"{base_path}/pretrained_weights"
# model_folder = f"{base_path}/echomimic_v2/pretrained_weights"
os.makedirs(model_folder, exist_ok=True)
download_model(
    repo_id="BadToBest/EchoMimicV2",
    target_folder=model_folder,
    ignore_patterns=["*.git*", "README.md"]
)

download_model(
    repo_id="stabilityai/sd-vae-ft-mse",
    target_folder=f"{model_folder}/sd-vae-ft-mse",
    ignore_patterns=["*.git*", "README.md"]
)

download_model(
    repo_id="lambdalabs/sd-image-variations-diffusers",
    target_folder=f"{model_folder}/sd-image-variations-diffusers",
    ignore_patterns=["*.git*", "README.md", "*.jpg"]
)

whisper_tiny_path = f"{model_folder}/audio_processor/tiny.pt"
whisper_tiny_url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
download_file(whisper_tiny_url, whisper_tiny_path)


ffmpeg_path = f"{base_path}/ffmpeg-4.4-amd64-static.tar.xz"
ffmpeg_url = "https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.4-amd64-static.tar.xz"
download_file(ffmpeg_url, ffmpeg_path)

import tarfile
with tarfile.open(ffmpeg_path, "r:xz") as tar:
    tar.extractall(f"{base_path}/")
if os.path.exists(f"{base_path}/ffmpeg-4.4-amd64-static"):
  os.rename(f"{base_path}/ffmpeg-4.4-amd64-static",f"{base_path}/ffmpeg")
