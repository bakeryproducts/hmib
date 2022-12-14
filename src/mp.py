from functools import partial

import fire
import numpy as np
from tqdm import tqdm
import rasterio as rio
import multiprocessing as mp

from block_utils import generate_block_coords, pad_block, mp_func_wrapper, chunkify, paste_crop


def read_tiff(name, idx, blocks_coords, pad_size, boundless=True):
    block_cd = blocks_coords[idx]
    y, x, h, w = pad_block(*block_cd, pad_size)
    window = ((y, y + h), (x, x + w))
    block = rio.open(name).read(window=window, boundless=boundless)
    return block, block_cd


def qread(q, name, idxs, reader, *args, **kwargs):
    for idx in idxs:
        r = reader(name, idx, *args, **kwargs)
        q.put(r)


def parallel_read(img_name, num_processes, window=None):
    fd = rio.open(img_name)
    H,W = fd.shape

    it = parallel_block_read(img_name, 512, 0, num_processes)
    image = np.zeros((3, H, W), dtype=fd.dtypes[0])
    for block, block_cd in it:
        paste_crop(image, block, block_cd, 0)
    if window is not None:
        x,y,w,h = window
        image = image[:, y:y+h, x:x+w]
    fd.close()
    return image


def parallel_block_read(img_name, block_size, pad_size, num_processes):
    process_data_len = 4
    qsize = process_data_len * num_processes

    m = mp.Manager()
    q = m.Queue(maxsize=qsize)
    H,W = rio.open(img_name).shape

    block_coords = list(generate_block_coords(H, W, block_size=(block_size,block_size)))
    total_blocks = len(block_coords)
    #print(f"{total_blocks=}")

    reader_args = [(q, img_name, idxs, read_tiff, block_coords, pad_size) for idxs in chunkify(range(total_blocks), process_data_len)]
    reader = partial(mp_func_wrapper, qread)

    pbar = tqdm(total=total_blocks)
    with mp.Pool(num_processes) as p:
        g = p.imap_unordered(reader, reader_args)
        for _ in g:
            while not q.empty():
                i = q.get()
                pbar.update()
                q.task_done()
                yield i
    pbar.close()


def run(img_name, block_size, pad_ratio, num_processes):
    it = parallel_block_read(img_name, block_size, pad_ratio, num_processes)
    for block, block_cd in it:
        pass


if __name__ == '__main__':
    fire.Fire(run)
