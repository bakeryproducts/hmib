import os
import os.path as osp

import fire
import requests as rq
from tqdm import tqdm


BASE_URL = "https://brd.nci.nih.gov/brd/imagedownload/"


def download(image_name, save_dir, exist_ok=True):
    image_file = osp.join(save_dir, f"{image_name}.svs")
    if osp.exists(image_file):
        if exist_ok:
            return
        raise FileExistsError(image_file)

    url = BASE_URL + image_name
    response = rq.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Unexpected response {response.status_code} for {image_name}")

    with open(image_file, "wb") as outf:
        outf.write(response.content)


def main(
    pieces_dir="input/extra/gtex/pieces",
    images_dir="input/extra/gtex/images",
):
    for organ in os.listdir(pieces_dir):
        organ_pieces_dir = osp.join(pieces_dir, organ)
        organ_images_dir = osp.join(images_dir, organ)
        for image_name in tqdm(os.listdir(organ_pieces_dir), desc=f"Downloading {organ}"):
            image_dir = osp.join(organ_pieces_dir, image_name)
            if not osp.isdir(image_dir):
                continue

            download(image_name, save_dir=organ_images_dir)


if __name__ == "__main__":
    fire.Fire(main)
