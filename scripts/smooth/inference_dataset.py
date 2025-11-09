import numpy as np
from torch.utils.data import Dataset
import pyvips
import polars as pl
import math
import scipy.signal
from scipy.signal.windows import triang


class VipsImagePatchedDataset(Dataset):
    # Ref:
    # https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/smooth_tiled_predictions.py
    # https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/inference/tiles.py
    # https://github.com/bnsreenu/python_for_microscopists/blob/master/229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
    #
    def __init__(self, vips_img, patch_size, stride, padding=0):
        self.vips_img = vips_img
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dframe = self._compute_patches()

    def __len__(self):
        return len(self.dframe)

    @classmethod
    def output_dframe(cls):
        return {
            "patch-id": pl.Int64,
            "patch-top-left-coord": pl.List(pl.Int64, strict=True, length=2),
        }

    def __getitem__(self, idx):
        patch_info = self.dframe.row(idx)
        x, y = patch_info[1]
        K = self.patch_size
        W = self.vips_img.width
        H = self.vips_img.height
        x_start = max(x, 0)
        x_end = min(x + K, W)
        y_start = max(y, 0)
        y_end = min(y + K, H)
        x_size = x_end - x_start
        y_size = y_end - y_start
        patch = np.zeros((K, K, self.vips_img.bands), dtype=np.uint8)
        if x_size > 0 and y_size > 0:
            img_patch = self.vips_img.crop(x_start, y_start, x_size, y_size).numpy()
            x_offset = x_start - x if x_start >= x else 0
            y_offset = y_start - y if y_start >= y else 0
            patch[y_offset : y_offset + y_size, x_offset : x_offset + x_size, :] = (
                img_patch
            )
        return patch_info[0], patch

    def _compute_patches(self):
        patch_id = 0
        patch_data = []
        W = self.vips_img.width
        H = self.vips_img.height
        K = self.patch_size
        S = self.stride
        P = self.padding
        n_patches_x = math.ceil((W + 2 * P - K) / S) + 1
        n_patches_y = math.ceil((H + 2 * P - K) / S) + 1
        x_positions = [i * S - P for i in range(n_patches_x)]
        y_positions = [j * S - P for j in range(n_patches_y)]
        for y in y_positions:
            for x in x_positions:
                patch_data.append(
                    {"patch-id": patch_id, "patch-top-left-coord": (x, y)}
                )
                patch_id += 1
        return pl.DataFrame(patch_data)

    def reconstruct_image(self):
        W = self.vips_img.width
        H = self.vips_img.height
        K = self.patch_size
        result = np.zeros((H, W, self.vips_img.bands), dtype=np.uint8)
        for row in self.dframe.iter_rows(named=True):
            patch_id = row["patch-id"]
            x, y = row["patch-top-left-coord"]
            x_start_img = max(x, 0)
            x_end_img = min(x + K, W)
            y_start_img = max(y, 0)
            y_end_img = min(y + K, H)
            x_size = x_end_img - x_start_img
            y_size = y_end_img - y_start_img
            patch = np.zeros((K, K, self.vips_img.bands), dtype=np.uint8)
            if x_size > 0 and y_size > 0:
                img_patch = self.vips_img.crop(
                    x_start_img, y_start_img, x_size, y_size
                ).numpy()
                x_offset = x_start_img - x if x_start_img >= x else 0
                y_offset = y_start_img - y if y_start_img >= y else 0
                patch[y_offset : y_offset + y_size, x_offset : x_offset + x_size, :] = (
                    img_patch
                )
            result[y_start_img:y_end_img, x_start_img:x_end_img, :] = patch[
                y_offset : y_offset + y_size, x_offset : x_offset + x_size, :
            ]
        return result

    @staticmethod
    def _spline_window(window_size, power=2):
        """
        Squared spline (power=2) window function.
        """
        intersection = int(window_size / 4)
        wind_outer = (
            abs(2 * (triang(window_size))) ** power
        ) / 2  # Updated to use triang from windows
        wind_outer[intersection:-intersection] = 0
        wind_inner = (
            1 - (abs(2 * (triang(window_size) - 1)) ** power) / 2
        )  # Updated here too
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0
        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind

    _cached_2d_windows = {}

    @staticmethod
    def _window_2D(window_size, power=2):
        """
        Make a 2D window function from a 1D spline window.
        """
        key = f"{window_size}_{power}"
        if key in VipsImagePatchedDataset._cached_2d_windows:
            wind = VipsImagePatchedDataset._cached_2d_windows[key]
        else:
            wind = VipsImagePatchedDataset._spline_window(window_size, power)
            wind = wind[:, None] * wind[None, :]  # Outer product for 2D
            VipsImagePatchedDataset._cached_2d_windows[key] = wind
        return wind

    @staticmethod
    def _rotate_mirror_do(pred):
        """
        Apply 8 rotations and mirrors to the prediction (shape: channels, K, K).
        """
        mirrs = []
        mirrs.append(pred.copy())
        mirrs.append(np.rot90(pred, axes=(1, 2), k=1))
        mirrs.append(np.rot90(pred, axes=(1, 2), k=2))
        mirrs.append(np.rot90(pred, axes=(1, 2), k=3))
        pred_mirror = pred[:, :, ::-1]
        mirrs.append(pred_mirror)
        mirrs.append(np.rot90(pred_mirror, axes=(1, 2), k=1))
        mirrs.append(np.rot90(pred_mirror, axes=(1, 2), k=2))
        mirrs.append(np.rot90(pred_mirror, axes=(1, 2), k=3))
        return mirrs

    @staticmethod
    def _rotate_mirror_undo(im_mirrs):
        """
        Undo rotations and mirrors, averaging the results.
        """
        origs = []
        origs.append(im_mirrs[0])
        origs.append(np.rot90(im_mirrs[1], axes=(1, 2), k=3))
        origs.append(np.rot90(im_mirrs[2], axes=(1, 2), k=2))
        origs.append(np.rot90(im_mirrs[3], axes=(1, 2), k=1))
        origs.append(im_mirrs[4][:, :, ::-1])
        origs.append(np.rot90(im_mirrs[5], axes=(1, 2), k=3)[:, :, ::-1])
        origs.append(np.rot90(im_mirrs[6], axes=(1, 2), k=2)[:, :, ::-1])
        origs.append(np.rot90(im_mirrs[7], axes=(1, 2), k=1)[:, :, ::-1])
        return np.mean(origs, axis=0)

    def reconstruct_maps(self, maps_list):
        """
        Reconstruct the full maps from patch predictions with smooth blending.

        Args:
            maps_list: List of dicts, each containing 'cellpose' and 'type' as numpy arrays
                       of shape (batch_size, channels, patch_size, patch_size)

        Returns:
            dict: {'cellpose': (2, H, W), 'type': (8, H, W)} where H and W are image dimensions
        """
        W = self.vips_img.width
        H = self.vips_img.height
        K = self.patch_size
        subdivisions = 2  # Matches your stride overlap
        window = self._window_2D(K, power=2)  # (K, K)

        result_cellpose = np.zeros((2, H, W), dtype=np.float32)
        result_type = np.zeros((8, H, W), dtype=np.float32)
        weights_cellpose = np.zeros((H, W), dtype=np.float32)
        weights_type = np.zeros((H, W), dtype=np.float32)

        patch_idx = 0
        for batch_maps in maps_list:
            batch_cellpose = batch_maps["cellpose"]  # (batch_size, 2, K, K)
            batch_type = batch_maps["type"]  # (batch_size, 8, K, K)

            # Ensure they are numpy arrays
            if hasattr(batch_cellpose, "cpu"):
                batch_cellpose = batch_cellpose.cpu().numpy()
            if hasattr(batch_type, "cpu"):
                batch_type = batch_type.cpu().numpy()

            batch_size = batch_cellpose.shape[0]

            for i in range(batch_size):
                if patch_idx >= len(self.dframe):
                    break

                patch_info = self.dframe.row(patch_idx)
                x, y = patch_info[1]

                # Compute the valid region in the image
                x_start = max(x, 0)
                x_end = min(x + K, W)
                y_start = max(y, 0)
                y_end = min(y + K, H)
                x_size = x_end - x_start
                y_size = y_end - y_start

                if x_size > 0 and y_size > 0:
                    x_offset = x_start - x if x_start > x else 0
                    y_offset = y_start - y if y_start > y else 0

                    cellpose_patch = batch_cellpose[i]  # (2, K, K)
                    rotated_cellpose = self._rotate_mirror_do(cellpose_patch)
                    weighted_rotated_cellpose = [
                        rot * window[None, :, :] for rot in rotated_cellpose
                    ]
                    averaged_cellpose = self._rotate_mirror_undo(
                        weighted_rotated_cellpose
                    )
                    valid_cellpose = averaged_cellpose[
                        :, y_offset : y_offset + y_size, x_offset : x_offset + x_size
                    ]
                    result_cellpose[:, y_start:y_end, x_start:x_end] += valid_cellpose
                    weights_cellpose[y_start:y_end, x_start:x_end] += window[
                        y_offset : y_offset + y_size, x_offset : x_offset + x_size
                    ]

                    type_patch = batch_type[i]  # (8, K, K)
                    rotated_type = self._rotate_mirror_do(type_patch)
                    weighted_rotated_type = [
                        rot * window[None, :, :] for rot in rotated_type
                    ]
                    averaged_type = self._rotate_mirror_undo(weighted_rotated_type)
                    valid_type = averaged_type[
                        :, y_offset : y_offset + y_size, x_offset : x_offset + x_size
                    ]
                    result_type[:, y_start:y_end, x_start:x_end] += valid_type
                    weights_type[y_start:y_end, x_start:x_end] += window[
                        y_offset : y_offset + y_size, x_offset : x_offset + x_size
                    ]

                patch_idx += 1

        mask_cellpose = weights_cellpose > 0
        result_cellpose[:, mask_cellpose] /= weights_cellpose[mask_cellpose]

        mask_type = weights_type > 0
        result_type[:, mask_type] /= weights_type[mask_type]

        return {"cellpose": result_cellpose, "type": result_type}
