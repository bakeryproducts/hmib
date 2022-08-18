import albumentations.augmentations.geometric.functional as AGF
import numpy as np
import rasterio as rio

from block_utils import (
    generate_block_coords,
    pad_block,
)


class TiffReader:
    """Reads tiff files.

    If subdatasets are available, then use them, otherwise just handle as usual.
    """

    def __init__(self, path_to_tiff_file: str):
        self.ds = rio.open(path_to_tiff_file)
        self.subds_list = [rio.open(subds_path) for subds_path in self.ds.subdatasets]

    def read(self, window=None, boundless=True):
        """
        Returns
        -------
        output: np.array of shape (height, width, channels)
            Result image
        """
        ds_kwargs = {}
        if window is not None:
            ds_kwargs.update({'window': window, 'boundless': boundless})

        if self.is_subsets_avail:
            output = np.vstack(
                [ds.read(**ds_kwargs) for ds in self.subds_list])
        else:
            output = self.ds.read(**ds_kwargs)

        output = output.transpose((1, 2, 0))
        return output

    def read_block(self, y, x, h, w, boundless=True):
        return self.read(
            window=((y, y + h), (x, x + w)),
            boundless=boundless
        )

    @property
    def is_subsets_avail(self):
        return len(self.subds_list) > 0

    @property
    def shape(self):
        if self.is_subsets_avail:
            return self.subds_list[0].shape
        else:
            return self.ds.shape

    def __del__(self):
        del self.ds
        del self.subds_list

    def close(self):
        self.ds.close()
        for subds in self.subds_list:
            subds.close()


class BatchedTiffReader(TiffReader):
    def __init__(
            self,
            path_to_tiff_file: str,
            block_size: int,
            network_scale: float,
            pad_ratio: float,
            batch_size: int
    ):
        super().__init__(path_to_tiff_file)

        self.block_size = block_size
        self.network_scale = network_scale
        self.pad_ratio = pad_ratio
        self.batch_size = batch_size

        height, width = self.shape
        scaled_block_size = self.scaled_block_size
        self.blocks_coords = list(generate_block_coords(
            height, width, block_size=(scaled_block_size, scaled_block_size)
        ))
        self.next_block = 0

    def __len__(self):
        return int(np.ceil(self.total_blocks / self.batch_size))

    @property
    def scaled_block_size(self):
        return int(round(self.block_size / self.network_scale))

    @property
    def pad_size(self):
        return int(round(self.block_size * self.pad_ratio))

    @property
    def scaled_pad_size(self):
        return int(round(self.scaled_block_size * self.pad_ratio))

    @property
    def total_blocks(self):
        return len(self.blocks_coords)

    @property
    def inv_network_scale(self):
        return 1.0 / self.network_scale

    def has_next_block(self):
        return self.next_block < len(self.blocks_coords)

    def read_batch(self):
        if not self.has_next_block():
            return None

        batch_blocks, batch_coords = [], []
        for i in range(self.batch_size):
            if not self.has_next_block():
                break

            block_cd = self.blocks_coords[self.next_block]
            padded_block_cd = pad_block(*block_cd, self.scaled_pad_size)
            block = self.read_block(*padded_block_cd)
            block = AGF.scale(block, self.network_scale)
            batch_blocks.append(block)
            batch_coords.append(block_cd)

            self.next_block += 1

        return np.stack(batch_blocks), np.stack(batch_coords)

    def __iter__(self):
        return iter(self.read_batch, None)


def load_tiff(tiff_file):
    reader = TiffReader(tiff_file)
    return reader.read()
