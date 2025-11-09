from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms.v2.functional import normalize, resize, to_dtype
from typing import Dict, List, Tuple
from skimage.filters import apply_hysteresis_threshold
from skimage.color import hsv2rgb
from scipy.ndimage import binary_fill_holes, find_objects, maximum_filter1d


def binarize(inst_map):
    binary = np.copy(inst_map > 0)
    return binary.astype("uint8")


def get_seeds(p, rpad=20, dims=2):
    shape = p.shape[1:]
    dims = len(p)
    pflows = []
    edges = []

    for i in range(dims):
        pflows.append(np.int32(p[i]).flatten())
        edges.append(np.arange(-0.5 - rpad, shape[i] + 0.5 + rpad, 1))  # bins

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)
    seeds: Tuple[np.ndarray, np.ndarray] = np.nonzero(
        np.logical_and(h - hmax > -1e-6, h > 10)
    )

    Nmax: np.ndarray = h[seeds]
    isort: np.ndarray = np.argsort(Nmax)[::-1]
    seeds = [s[isort] for s in seeds]

    return seeds, h, pflows


def expand_seed_pixels(seeds, h, dims=2):
    pix = list(np.array(seeds).T)
    shape = h.shape
    expand = np.nonzero(np.ones((3, 3)))
    for k in range(len(pix)):
        pix[k] = list(pix[k])

        newpix = []
        iin = []
        for i, e in enumerate(expand):
            epix: np.ndarray = (
                e[:, None] + np.expand_dims(pix[k][i], 0) - 1
            )  # (9, n_pixels)
            epix: np.ndarray = epix.flatten()
            iin.append(np.logical_and(epix >= 0, epix < shape[i]))
            newpix.append(epix)

        iin = np.all(tuple(iin), axis=0)
        newpix = [pi[iin] for pi in newpix]

        newpix = tuple(newpix)
        igood = h[newpix] > 2
        for i in range(dims):
            pix[k][i] = newpix[i][igood]

        pix[k] = tuple(pix[k])

    return pix


def get_masks_cellpose(p, rpad=20):
    shape0 = p.shape[1:]
    dims = len(p)

    seeds, h, pflows = get_seeds(p, rpad, dims)
    pix = expand_seed_pixels(seeds, h, dims)

    M = np.zeros(h.shape, np.int32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k

    for i in range(dims):
        pflows[i] = pflows[i] + rpad

    M0 = M[tuple(pflows)]
    _, counts = np.unique(M0, return_counts=True)
    big = float(np.prod(shape0))
    for i in np.nonzero(counts > big)[0]:
        M0[M0 == i] = 0

    _, M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    return M0


def post_proc_cellpose(
    inst_map,
    flow_map,
    dist_map=None,
    return_flows=False,
    min_size=30,
    interp=True,
    use_gpu=True,
    **kwargs,
):
    #  convert channels to CHW
    if dist_map is not None:
        binary_mask = apply_hysteresis_threshold(dist_map, 0.5, 0.5)
    else:
        binary_mask = binarize(inst_map).astype(bool)

    dP = flow_map * binary_mask
    # dP = normalize_field(dP)

    pixel_loc, _ = follow_flows(
        dP,
        niter=200,
        mask=binary_mask,
        suppress_euler=False,
        interp=interp,
        use_gpu=use_gpu,
    )

    mask = get_masks_cellpose(pixel_loc)
    inst_map = fill_holes_and_remove_small_masks(mask, min_size=min_size).astype("i4")

    if return_flows:
        hsv_flows = gen_flows(dP)
        return inst_map, hsv_flows

    return inst_map


def steps2D_interp(p, dP, niter=200, suppress_euler=False, use_gpu=True):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    shape = dP.shape[1:]
    shape = np.array(shape)[[1, 0]].astype("double") - 1

    pt = torch.from_numpy(p[[1, 0]].T).double().to(device)
    pt = pt.unsqueeze(0).unsqueeze(0)

    im = torch.from_numpy(dP[[1, 0]]).double().to(device)
    im = im.unsqueeze(0)

    for k in range(2):
        im[:, k, ...] *= 2.0 / shape[k]
        pt[..., k] /= shape[k]

    pt = pt * 2 - 1

    for t in range(niter):
        dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)

        if suppress_euler:
            dPt /= 1 + t

        # clamp the final pixel locations
        for k in range(2):
            pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k, ...], -1.0, 1.0)

    pt = (pt + 1) * 0.5
    for k in range(2):
        pt[..., k] *= shape[k]

    return pt[..., [1, 0]].cpu().numpy().squeeze().T


def steps2D(p, dP, inds, niter=200, suppress_euler=False):
    shape = p.shape[1:]
    for t in prange(niter):
        for j in range(inds.shape[0]):
            y = inds[j, 0]
            x = inds[j, 1]
            p0, p1 = int(p[0, y, x]), int(p[1, y, x])
            step = dP[:, p0, p1]

            if suppress_euler:
                step /= 1 + t

            for k in range(p.shape[0]):
                p[k, y, x] = min(shape[k] - 1, max(0, p[k, y, x] + step[k]))

    return p


def percentile_normalize(img, lower=0.01, upper=99.99):
    axis = (0, 1)

    if img.ndim not in (2, 3):
        raise ValueError(
            f"Input img needs to have shape (H, W, C)|(H, W). Got: {img.shape}"
        )
    im = img.copy()
    upercentile = np.percentile(im, upper)
    lpercentile = np.percentile(im, lower)

    return np.interp(im, (lpercentile, upercentile), axis).astype(np.float32)


def percentile_normalize99(img, amin=None, amax=None):
    axis = (0, 1)
    if img.ndim not in (2, 3):
        raise ValueError(
            f"Input img needs to have shape (H, W, C)|(H, W). Got: {img.shape}"
        )

    im = img.copy()
    percentile1 = np.percentile(im, q=1, axis=axis)
    percentile99 = np.percentile(im, q=99, axis=axis)
    im = (im - percentile1) / (percentile99 - percentile1 + 1e-7)
    if not any(x is None for x in (amin, amax)):
        im = np.clip(im, a_min=amin, a_max=amax)

    return im.astype(np.float32)


def follow_flows(
    dP, mask=None, niter=200, suppress_euler=False, interp=True, use_gpu=True
):
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)

    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    p = np.array(p).astype(np.float32)

    seeds = np.abs(dP[0]) > 1e-3
    if mask is not None:
        seeds = np.logical_or(mask, seeds)

    pixel_loc = np.nonzero(seeds)
    inds = np.array(pixel_loc).astype(np.int32).T

    if interp:
        try:
            p[:, inds[:, 0], inds[:, 1]] = steps2D_interp(
                p=p[:, inds[:, 0], inds[:, 1]],
                dP=dP,
                niter=niter,
                suppress_euler=suppress_euler,
                use_gpu=use_gpu,
            )
        except Exception:
            pass
    else:
        try:
            p = steps2D(p, dP.astype(np.float32), inds, niter)
        except Exception:
            pass

    return p, inds


def gen_flows(hover):
    enhanced = percentile_normalize99(hover, amin=-1, amax=1)
    H = (np.arctan2(enhanced[0], enhanced[1]) + np.pi) / (2 * np.pi)
    S = percentile_normalize(enhanced[0] ** 2 + enhanced[1] ** 2)
    HSV = np.stack([H, S, S], axis=-1)
    HSV = np.clip(HSV, a_min=0.0, a_max=1.0)
    flow = (hsv2rgb(HSV) * 255).astype(np.uint8)

    return flow


def fill_holes_and_remove_small_masks(inst_map, min_size: int = 15):
    if inst_map.ndim != 2:
        raise ValueError(f"`inst_map` shape need to be 2D. Got {inst_map.shape}.")

    j = 0
    slices = find_objects(inst_map)
    for i, slc in enumerate(slices):
        if slc is not None:

            msk = inst_map[slc] == (i + 1)
            npix = msk.sum()

            if min_size > 0 and npix < min_size:
                inst_map[slc][msk] = 0
            else:
                msk = binary_fill_holes(msk)
                inst_map[slc][msk] = j + 1
                j += 1

    return inst_map


def majority_vote(type_map, inst_map):
    type_map = binarize(inst_map) * type_map
    pred_id_list = np.unique(inst_map)[1:]
    for inst_id in pred_id_list:
        inst_type = type_map[inst_map == inst_id]
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        cell_type = type_list[0][0]

        if cell_type == 0:
            if len(type_list) > 1:
                cell_type = type_list[1][0]

        type_map[inst_map == inst_id] = cell_type

    return type_map


def get_inst_map(type_prob_map, cellpose_map, **kwargs):
    return post_proc_cellpose(np.argmax(type_prob_map, axis=0), cellpose_map, **kwargs)


def get_type_map(prob_map, inst_map):
    return majority_vote(np.argmax(prob_map, axis=0), inst_map)


def post_proc_pipeline_single(maps, **kwargs):
    result = {}
    result["inst"] = get_inst_map(maps["type"], maps["cellpose"], **kwargs)
    result["type"] = get_type_map(maps["type"], result["inst"])
    result["inst"] *= result["type"] > 0
    return result


def post_proc_pipeline(maps, **kwargs):
    batch_size = maps["cellpose"].shape[0]
    result = {"inst": [], "type": []}
    for i in range(batch_size):
        single_map = {"cellpose": maps["cellpose"][i], "type": maps["type"][i]}
        res_inst = get_inst_map(single_map["type"], single_map["cellpose"], **kwargs)
        res_type = get_type_map(single_map["type"], res_inst)
        res_inst *= res_type > 0
        result["inst"].append(res_inst)
        result["type"].append(res_type)
    result["inst"] = np.stack(result["inst"], axis=0)
    result["type"] = np.stack(result["type"], axis=0)
    return result


class CellPoseProcessor:
    def __init__(
        self,
        model_path,
        model_input_shape=(224, 224),  # width, height
        mean=(0.707223, 0.578729, 0.703617),
        std=(0.211883, 0.230117, 0.177517),
        resize_output_to=(224, 224),
        use_gpu=True,
    ):
        self.model = torch.export.load(Path(model_path))
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.use_gpu = use_gpu

        self.mean = mean
        self.std = std
        self.model_input_shape = model_input_shape
        self.clamp = False

        self.resize_output_to = resize_output_to

    def preprocess_imgs(self, x):
        """
        Input:
            x: np.ndarray, shape (batch_size, H, W, C)
        Output:
            x: torch.tensor, shape (batch_size, C, H, W)
        """
        x = to_dtype(
            to_dtype(x.permute(0, 3, 1, 2), torch.uint8), torch.float32, scale=True
        )
        x = normalize(x, mean=self.mean, std=self.std)
        if self.clamp:
            x = torch.clamp(x, 0.0, 1.0)
        if tuple(x.shape[-2:]) != self.model_input_shape:
            x = resize(x, self.model_input_shape)
        return x

    def __call__(self, x):
        x = self.preprocess_imgs(x)
        if x.device != self.device:
            x = x.to(self.device)

        with torch.no_grad():
            maps = self.model.module()(x)

        if x.shape[-2:] != self.resize_output_to:
            maps["type"] = resize(maps["type"], self.resize_output_to)
            maps["cellpose"] = resize(maps["cellpose"], self.resize_output_to)

        if self.device != "cpu":
            maps = {k: v.cpu().numpy() for k, v in maps.items()}
        else:
            maps = {k: v.numpy() for k, v in maps.items()}

        return maps
