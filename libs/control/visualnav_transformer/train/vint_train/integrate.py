# %%
import os
import pickle
import sys
from typing import Any, Dict
import numpy as np
import networkx as nx
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import time
import h5py
import lmdb

import torch
import torch.nn.functional as F


def normalize_pls(pls, scale_factor=100, outlier_value=99, new_max_val=None):

    outliers = pls >= outlier_value
    # if all are outliers, set them to zero
    if sum(outliers) == len(pls):
        return np.zeros_like(pls)

    min_val = pls.min()
    if new_max_val is None:
        new_max_val = pls[~outliers].max() + 1
    else:
        assert new_max_val > pls[~outliers].max(), f"{new_max_val} <= {pls[~outliers].max()}"

    # else set outliers to max value of inliers + 1
    # so that when normalized, they are set to 0
    if sum(outliers) > 0:
        pls[outliers] = new_max_val

    # include a dummy value to ensure that new_max_val -> 0 after norm 'even for inliers'
    pls = np.concatenate([pls, [new_max_val]])

    # normalize so that outliers are set to 0; inliers \in (0, scale_factor]
    pls = scale_factor * (new_max_val - pls) / (new_max_val - min_val)
    return pls[:-1]

def normalize_pls_pixels(pls_pixels, boost_final_goal, scale_factor=100, outlier_value=99, new_max_val=None):
    # TODO: Do we need epsilon here?
    is_final_goal = pls_pixels == 0

    outliers = pls_pixels >= outlier_value
    # if all are outliers, set them to zero
    if np.all(outliers):
        return np.zeros_like(pls_pixels)

    min_val = pls_pixels.min()
    if new_max_val is None:
        new_max_val = pls_pixels[~outliers].max() + 1
    else:
        assert (
            new_max_val > pls_pixels[~outliers].max()
        ), f"{new_max_val} <= {pls_pixels[~outliers].max()}"

    # else set outliers to max value of inliers + 1
    # so that when normalized, they are set to 0
    if np.any(outliers):
        pls_pixels[outliers] = new_max_val

    # normalize so that outliers are set to 0; inliers \in (0, scale_factor]
    pls_pixels = scale_factor * (new_max_val - pls_pixels) / (new_max_val - min_val)

    if boost_final_goal:
        # Reduce all pls_pixels by 10% (increasing their cost):
        pls_pixels *= 0.9
        # But set final goals to the maximum possible value:
        pls_pixels[is_final_goal] = scale_factor

    return pls_pixels

def get_masks_gradient(masks):
    """
    masks: [N, H, W]
    """
    dx = np.zeros_like(masks)
    dy = dx.copy()
    masks_f = masks.copy().astype(float)

    dx[:, 1:, :] = abs(masks_f[:, 1:, :] - masks_f[:, :-1, :])
    dy[:, :, 1:] = abs(masks_f[:, :, 1:] - masks_f[:, :, :-1])

    grad = ((dx + dy).sum(0).astype(bool)).astype(float)
    return grad


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def generate_positional_encodings(max_rank, d_model):
    """
    Generate positional encodings for ranks from 0 to max_rank.
    
    Parameters:
    max_rank (int): The maximum rank for which to generate encodings.
    d_model (int): The dimensionality of the encoding vector.

    Returns:
    dict: A dictionary with ranks as keys and positional encoding vectors as values.

    # Example usage:
    max_rank = 5
    d_model = 16
    positional_encodings = generate_positional_encodings(max_rank, d_model)
    print(positional_encodings)
    """
    reduceLater = False
    reduceLaterDim = d_model
    if d_model < 4:
        reduceLater = True
        d_model = 4
    ranks = np.arange(max_rank + 1)
    encodings = np.zeros((max_rank + 1, d_model))
    
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    encodings[:, 0::2] = np.sin(ranks[:, np.newaxis] * div_term)
    encodings[:, 1::2] = np.cos(ranks[:, np.newaxis] * div_term)
    if reduceLater:
        encodings = encodings[:,:reduceLaterDim]
    return encodings


def random_crop_and_reshape_torch(masks_np, mask_crop_ratio=0.8):
    """
    Randomly crops binary masks with separate max crop % for height and width,
    then resizes back to original size. Aspect ratio can change.

    Args:
        masks_np (np.ndarray): Binary masks of shape (H, W, D)
        mask_crop_ratio (float): Max crop percent for height and width.

    Returns:
        np.ndarray: Resized masks of shape (H, W, D)
    """
    H, W, D = masks_np.shape

    if not (0 < mask_crop_ratio <= 1.0):
        raise ValueError("Crop percentages must be in (0, 1]")

    # Random crop percentages (from [max_pct, 1.0])
    crop_pct_h = np.random.uniform(mask_crop_ratio, 1.0)
    crop_pct_w = np.random.uniform(mask_crop_ratio, 1.0)

    crop_h = int(H * crop_pct_h)
    crop_w = int(W * crop_pct_w)

    top = np.random.randint(0, H - crop_h + 1)
    left = np.random.randint(0, W - crop_w + 1)

    # Torch: (D, 1, H, W)
    masks = torch.from_numpy(masks_np).permute(2, 0, 1).unsqueeze(1).float()

    # Crop and resize
    cropped = masks[:, :, top:top + crop_h, left:left + crop_w]
    resized = F.interpolate(cropped, size=(H, W), mode='nearest')

    return resized.squeeze(1).permute(1, 2, 0).to(torch.uint8).numpy()


class TopoPaths:
    def __init__(self,inPath,datasetName,maxRank=200,dims=16,w=160,h=120,dims_segFt=None,goal_use_pl=0, precomputed_filename=None, pl_perturb_ratio=0.0, pl_perturb_type="max_val", mask_crop_ratio=1.0, use_mask_grad=False):
        self.inPath = inPath
        self.datasetName = datasetName
        self.readPath = f"{inPath}/{datasetName}/"
        self.dims_segFt = dims_segFt
        self.w, self.h = w, h
        self.pl_outlier_value = 99
        self.pl_perturb_ratio = pl_perturb_ratio
        self.pl_perturb_type = pl_perturb_type
        self.mask_crop_ratio = mask_crop_ratio
        self.use_mask_grad = use_mask_grad

        print("Loading precomputed masks and pls...")
        # TODO: check compat with go_stanford
        fname = precomputed_filename
        if goal_use_pl == 0:
            if self.datasetName == "hm3d_iin_train":
                raise ValueError("goal_use_pl=0 is not supported for hm3d_iin_train")
        self.masks_pls_dict_path = f"{self.inPath}/{self.datasetName}{fname}"
        if 'gt_topometric' in fname:
            self.pl_outlier_value = 256 # bc it is already scaled 0 to 255
        elif 'e3d_' in fname:
            self.pl_outlier_value = 255 # max_val for new runs to be 255 instead of 100

        # TODO: segment features currently not being used
        if type(self.dims_segFt) == int:
            print("Loading precomputed segment features...")
            segFt_dict_path = f"{self.inPath}/{self.datasetName}_segFt_pca_trajDict.npz"
            segFt_dict = np.load(segFt_dict_path,allow_pickle=True)
            self.segFt_dict = {}
            for k in segFt_dict:
                try:
                    self.segFt_dict[k] = segFt_dict[k][()]
                except:
                    pass
            print("Done!")

        if dims == 1:
            self.rank_enc = np.arange(maxRank+1).astype(float).reshape(-1,1)
        else:
            self.rank_enc = generate_positional_encodings(maxRank, dims)
        self.default_enc = np.ones((dims, self.h//2, self.w//2)) * self.rank_enc[0][:,None,None] # (D,H,W)
        if type(self.dims_segFt) == int:
            self.default_enc2 = np.ones((self.dims_segFt,h//2,w//2))

    def load_segFt(self,trajName):
        segFt = None
        if trajName in self.segFt_dict:
            segFt = self.segFt_dict[trajName]
        return segFt

    def perturb_mask_pls(self, pls):
        perturbed_pls = pls.copy()
        if self.pl_perturb_ratio == 0:
            return pls

        num_masks_to_perturb = int(len(pls) * self.pl_perturb_ratio)
        indices_to_perturb = np.random.choice(len(pls), num_masks_to_perturb, replace=False)

        if self.pl_perturb_type == "max_val":
            perturbed_pls[indices_to_perturb] = self.pl_outlier_value
        elif self.pl_perturb_type == "rand_from_inliers":
            inliers = pls[pls < self.pl_outlier_value]
            perturbed_pls[indices_to_perturb] = np.random.choice(inliers, num_masks_to_perturb)
        else:
            raise ValueError(f"Invalid pl_perturb_type {self.pl_perturb_type}")

        return perturbed_pls

    def get_topo_path(self, trajName, imgIdx, getFt=False, goalIdx=None):
        segFt = None
        t0 = time.time()
        key = f"{trajName}_{imgIdx}"
        with h5py.File(self.masks_pls_dict_path, "r") as masks_pls_dict:
            # print(time.time()-t0,"Checking if trajName exists in masks_pls_dict...")
            if key not in masks_pls_dict:
                return self.create_input(None,None)
            else:
                key_data = masks_pls_dict[key]
                # print(time.time()-t0, "Loading masks and pls...")
                img_size = key_data["size"][()]
                img_masks = key_data['img_masks']
                # read masks in order
                img_masks = [{"size": img_size, "counts": img_masks[f"{mi}"][()]} for mi in range(len(img_masks.keys()))]
                if goalIdx is not None:
                    img_pls = key_data['img_pls_allCenterGoals'][()]
                    goalIdx_ = goalIdx-imgIdx
                    # check if goal_is_negative
                    if goalIdx_ < 0 or goalIdx_ >= img_pls.shape[1]:
                        goalIdx_ = -1
                    img_pls = img_pls[:,goalIdx_]
                else:
                    img_pls = key_data['img_pls'][()]
                img_pls = self.perturb_mask_pls(img_pls)
                # print(time.time()-t0, "Loading segment features...")
                if getFt:
                    segFt = self.load_segFt(trajName)
                    if segFt is not None:
                        segFt = segFt[imgIdx].T # DxS
                # print(time.time()-t0, "Creating input...")
            inputData = self.create_input(img_pls,img_masks,convertMask=True,segFt=segFt,t0=t0)
        return inputData

    def create_input(self,pls,masks,convertMask=False,segFt=None,t0=0):
        img_enc = self.default_enc if segFt is None else self.default_enc2
        plWtColorImg = np.zeros((3,img_enc.shape[1],img_enc.shape[2]))
        if pls is None or masks is None:
            pass
        else:
            pls = normalize_pls(pls, outlier_value=self.pl_outlier_value)
            # pls = normalize_pls(pls.copy(), scale_factor=100, outlier_value=self.pl_outlier_value, new_max_val=self.pl_outlier_value)
            # print(time.time()-t0, "Converting masks to image...")
            if convertMask:
                masks = np.array([rle_to_mask(m) for m in masks])
            masks = masks.transpose([1,2,0])[::2,::2] # (H,W,D)
            # for topological graphs of masks 320, 240
            if masks.shape[0] != self.h//2:
                masks = masks[::2, ::2]
            if masks.shape[0] != self.h//2:
                raise ValueError(f"masks shape {masks.shape} does not match expected shape ({self.h//2},{self.w//2})")

            # randomly crop masks
            if self.mask_crop_ratio != 1.0:
                masks = random_crop_and_reshape_torch(masks, self.mask_crop_ratio)

            deno = masks.sum(-1)
            deno[deno == 0] = 1
            # print(time.time()-t0, "Computing colors...")
            colors, norm = value2color(pls, cmName='winter')
            plWtColorImg = (masks/deno[:,:,None] @ colors).transpose(2,0,1)
            # print(time.time()-t0, "Computing image encodings...")
            if segFt is not None:
                img_enc = (masks @ segFt.T[:,:self.dims_segFt]).transpose(2,0,1)/deno[None,:,:] # (D,H,W)
            else:
                enc = self.rank_enc[pls.astype(int)]
                img_enc = (masks @ enc).transpose(2,0,1)
        # if img_enc.shape[0] >= 3:
        #     plWtColorImg = img_enc[:3,:,:]
        # else:
        #     plWtColorImg = np.stack([img_enc[0]]*3)
        # print(time.time()-t0, "returning img_enc, plWtColorImg")
        if img_enc.dtype == object:
            print("img",img_enc.shape)
        if plWtColorImg.dtype == object:
            print("plt",plWtColorImg.shape)
        if self.use_mask_grad:
            grad = get_masks_gradient(masks.transpose(2,0,1))
            img_enc = np.concatenate([img_enc, grad[None]], axis=0)
        # assert(img_enc.dtype != object and plWtColorImg.dtype != object)
        return img_enc, plWtColorImg

    def get_traj_names(self):
        return natsorted(os.listdir(self.readPath))

    # TODO: only for fname = "_masks_pls_allCenterGoals_trajDict.npz"
    # massive storage requirements
    def _build_cache(self):
        """
        Build a cache of goal images for faster loading using LMDB
        """
        cache_filename = f"{self.masks_pls_dict_path}.lmdb"

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for ti, trajName in enumerate(tqdm(self.masks_pls_dict)):
                        for imgIdx in range(len(self.masks_pls_dict[trajName]['img_masks'])):
                            goal_image, goal_vis = self.get_topo_path(trajName, imgIdx)
                            key = f"{trajName}_{imgIdx}"
                            # binary encode
                            goal_image = goal_image.tobytes()
                            goal_vis = goal_vis.tobytes()
                            txn.put(key.encode("ascii"), goal_image)
                            txn.put(f"{key}_vis".encode("ascii"), goal_vis)

                            ## TODO: instead store the input for create_input?
                            # img_masks = self.masks_pls_dict[trajName]['img_masks'][imgIdx]
                            # img_pls = self.masks_pls_dict[trajName]['img_pls'][imgIdx]

                        # Commit every 100 iterations to reduce RAM usage
                        if ti % 100 == 0 and ti > 0:
                            txn.commit()
                            txn = image_cache.begin(write=True)  # Start a new transaction
                    txn.commit()

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

def value2color(values,vmin=None,vmax=None,cmName='jet'):
    cmapPaths = matplotlib.cm.get_cmap(cmName)
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = np.array([cmapPaths(norm(value))[:3] for value in values])
    return colors, norm



# %%
## EXAMPLE 1 (go_stanford)
# homedir = os.path.expanduser("~")
# tp = TopoPaths(f"{homedir}/workspace/s/sg_habitat/out/RoboHop/","go_stanford")
# trajNames = tp.get_traj_names()
# imgEnc = tp.get_topo_path(trajNames[0],0)

## EXAMPLE 2 (hm3d)
# homedir = os.path.expanduser("~")
# tp = TopoPaths(f"{homedir}/fastdata/navigation/","hm3d_iin_train")
# trajNames = tp.get_traj_names()
# %%
