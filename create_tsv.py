import os
from pathlib import Path
from glob import glob
import pandas as pd

image_dir = "imgs"
# image_list = list(glob(Path(image_dir).joinpath("*").with_suffix(".jpg")))
ext_list = ["png", "jpg", "jpeg", "bmp", "gif"]
image_list = sum([glob(os.path.join(image_dir, f"*.{e}")) for e in ext_list], [])
image_data = pd.DataFrame(
    {
        "filename": image_list,
        "prediction": "P",
        # "tag": ("UNK" for s in enumerate(image_list)),
        "conf. score": (f"example_{i:0>3}" for i, s in enumerate(image_list)),
    }
)

filename = "image_list.txt"

image_data.to_csv(filename, sep="\t", index=False, header=False)

print(f"Write an image list to: {filename}")
