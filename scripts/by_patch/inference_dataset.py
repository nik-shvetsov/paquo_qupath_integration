import numpy as np
from torch.utils.data import Dataset
import math
import polars as pl


class VipsImagePatchedDataset(Dataset):
    def __init__(self, vips_img, patch_size):
        self.border_overlapping = False

        self.vips_img = vips_img
        self.patch_size = patch_size
        self.dframe = self._compute_patches()

    def __len__(self):
        return len(self.dframe)

    def __getitem__(self, idx):
        patch_info = self.dframe.row(idx)
        # patch_info[0] - "patch-id"
        # patch_info[1] - "patch-top-left-coord"
        patch = self.vips_img.crop(*patch_info[1], self.patch_size, self.patch_size)
        # patch = patch.resize(inference_resize_to, kernel='cubic')
        return patch_info[0], patch.numpy()

    def _compute_patches(self):
        patch_id = 0
        patch_data = []
        if self.border_overlapping:
            # Calculate the number of patches in x and y directions
            n_patches_x = math.ceil(self.vips_img.width / self.patch_size)
            n_patches_y = math.ceil(self.vips_img.height / self.patch_size)
        else:
            # Calculate the number of non-overlapping patches
            n_patches_x = self.vips_img.width // self.patch_size
            n_patches_y = self.vips_img.height // self.patch_size

        # Generate x positions
        x_positions = []
        for i in range(n_patches_x):
            x = i * self.patch_size
            x_positions.append(x)

        # Generate y positions
        y_positions = []
        for j in range(n_patches_y):
            y = j * self.patch_size
            y_positions.append(y)

        # Create patch data with adjusted coordinates
        for y in y_positions:
            for x in x_positions:
                patch_data.append(
                    {"patch-id": patch_id, "patch-top-left-coord": (x, y)}
                )
                patch_id += 1

        # Handle remaining area with padding if non-overlapping and not fully covered
        if not self.border_overlapping:
            if self.vips_img.width % self.patch_size != 0:
                x = n_patches_x * self.patch_size
                for y in y_positions:
                    patch_data.append(
                        {"patch-id": patch_id, "patch-top-left-coord": (x, y)}
                    )
                    patch_id += 1

            if self.vips_img.height % self.patch_size != 0:
                y = n_patches_y * self.patch_size
                for x in x_positions:
                    patch_data.append(
                        {"patch-id": patch_id, "patch-top-left-coord": (x, y)}
                    )
                    patch_id += 1

            # Handle bottom-right corner if both width and height are not fully covered
            if (self.vips_img.width % self.patch_size != 0) and (
                self.vips_img.height % self.patch_size != 0
            ):
                x = n_patches_x * self.patch_size
                y = n_patches_y * self.patch_size
                patch_data.append(
                    {"patch-id": patch_id, "patch-top-left-coord": (x, y)}
                )
                patch_id += 1

        return pl.DataFrame(patch_data)

    def reconstruct_maps(self, maps):
        result = np.zeros(
            (self.vips_img.height, self.vips_img.width, 3), dtype=np.uint8
        )
        for row in self.dframe.iter_rows(named=True):
            patch_id = row["patch-id"]
            x, y = row["patch-top-left-coord"]
            patch = maps[patch_id]
            x_end = min(x + self.patch_size, self.vips_img.width)
            y_end = min(y + self.patch_size, self.vips_img.height)
            h, w = y_end - y, x_end - x
            result[y:y_end, x:x_end, :] = patch[:h, :w, :]
        return result

    def reconstruct_image(self):
        result = np.zeros(
            (self.vips_img.height, self.vips_img.width, 3), dtype=np.uint8
        )
        for row in self.dframe.iter_rows(named=True):
            patch_id = row["patch-id"]
            x, y = row["patch-top-left-coord"]
            patch = self.vips_img.crop(x, y, self.patch_size, self.patch_size).numpy()
            x_end = min(x + self.patch_size, self.vips_img.width)
            y_end = min(y + self.patch_size, self.vips_img.height)
            h, w = y_end - y, x_end - x
            result[y:y_end, x:x_end, :] = patch[:h, :w, :]
        return result
