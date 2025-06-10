import os
import urllib.request

url_base = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
files = {
    "t10k-images-idx3-ubyte.gz": "data/fashion/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "data/fashion/t10k-labels-idx1-ubyte.gz",
}

os.makedirs("data/fashion", exist_ok=True)

for filename, out_path in files.items():
    if not os.path.exists(out_path.replace(".gz", "")):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url_base + filename, out_path)
        os.system(f"gzip -d {out_path}")
print("✅ 測試資料下載完成")
