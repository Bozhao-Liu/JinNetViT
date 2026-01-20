import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import os
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
import json
import shutil

eps = 1e-8

def mask_iou(mask1, mask2):
    """
    Compute IoU between two batches of binary masks.
    mask1, mask2: torch tensors of shape (N, H, W)
    Returns: tensor of shape (N,) with IoU per mask
    """
    mask1 = mask1.bool()
    mask2 = mask2.bool()
    inter = torch.logical_and(mask1, mask2).flatten(1).sum(dim=1).float()  # (N,)
    union = torch.logical_or(mask1, mask2).flatten(1).sum(dim=1).float()  # (N,)

    return inter / (union + eps)

from scipy.ndimage import binary_erosion, distance_transform_edt

def _ensure_single_mask(mask):
    """
    Ensures mask is a single 2D (H,W) boolean array.
    If mask is flattened (H*W,), this function CANNOT guess H,W.
    So it raises an error UNLESS you provide target shape.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)

    # Case 1: mask is flattened
    if mask.ndim == 1:
        raise ValueError(
            f"Mask is flattened to 1D with shape {mask.shape}. "
            f"Boundary metrics require the original 2D shape (H,W). "
            f"Please pass unflattened masks to compute_boundary()."
        )

    # Remove leading singleton dims
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask.squeeze(0)

    # Final check
    if mask.ndim != 2:
        raise ValueError(
            f"Expected mask to be (H,W), got shape {mask.shape}"
        )

    return mask.astype(bool)


def _extract_boundary(mask, iterations=1):
    structure = np.ones((3, 3), bool)
    eroded = binary_erosion(
        mask,
        structure=structure,
        iterations=iterations,
        border_value=0
    )
    return mask ^ eroded


def _surface_distances(gt, pr):
    """
    Returns a 1D array of symmetric surface distances.
    NaN indicates invalid configurations (e.g., one empty mask).
    """
    # Both completely empty: no meaningful contour – let caller decide
    if not gt.any() and not pr.any():
        return np.array([np.nan])

    # One empty, one non-empty → undefined symmetric surface distance
    if gt.any() and not pr.any():
        return np.array([np.nan])

    if pr.any() and not gt.any():
        return np.array([np.nan])

    # Compute boundaries
    surf_gt = _extract_boundary(gt)
    surf_pr = _extract_boundary(pr)

    # If either boundary is empty, distances are not well-defined
    if not surf_gt.any() or not surf_pr.any():
        return np.array([np.nan])

    # Distance transforms
    dt_pr = distance_transform_edt(~pr)
    dt_gt = distance_transform_edt(~gt)

    d1 = dt_pr[surf_gt]
    d2 = dt_gt[surf_pr]

    if d1.size + d2.size == 0:
        return np.array([np.nan])

    d = np.concatenate([d1, d2])
    return d


# ============================================================
# PUBLIC FUNCTION — USE THIS
# ============================================================

def compute_boundary_metrics_single(gt_mask, pred_mask):
    """
    Compute BIoU, HD95, and MSD for a SINGLE 2D mask pair.
    No batches allowed.

    Returns:
        dict with keys:
            'BIoU'
            'HD95'
            'MSD'

    Convention:
    - Completely empty GT and prediction (no object at all) → all metrics NaN (ignored later).
    - Degenerate boundary cases (no boundary pixels) → all metrics NaN.
    """

    gt = _ensure_single_mask(gt_mask)
    pr = _ensure_single_mask(pred_mask)

    # If both masks are completely empty (no object), skip this image in boundary stats.
    if not gt.any() and not pr.any():
        return {
            "BIoU": np.nan,
            "HD95": np.nan,
            "MSD": np.nan
        }

    # --- BIoU ---
    gt_b = _extract_boundary(gt)
    pr_b = _extract_boundary(pr)

    inter = np.logical_and(gt_b, pr_b).sum()
    union = np.logical_or(gt_b, pr_b).sum()

    # If union is 0, there are no boundary pixels in either GT or prediction.
    # This is a degenerate case for boundary IoU → treat as invalid (NaN) and skip.
    if union == 0:
        biou = np.nan
    else:
        biou = inter / (union + 1e-8)

    # --- Distances ---
    d = _surface_distances(gt, pr)
    d_valid = d[~np.isnan(d)]

    if d_valid.size == 0:
        hd95 = np.nan
        msd = np.nan
    else:
        hd95 = float(np.percentile(d_valid, 95))
        msd = float(np.mean(d_valid))

    return {
        "BIoU": float(biou) if not np.isnan(biou) else np.nan,
        "HD95": hd95,
        "MSD": msd
    }


def compute_boundary(gt_masks, pred_masks):
    """
    Accepts:
        - (N, H, W)
        - (N, 1, H, W)
        - (N, H*W)   -> auto-reshapes to (N, H, W)

    Auto-detects H,W from flattened size.
    """

    # Convert to numpy if needed
    if isinstance(gt_masks, torch.Tensor):
        gt_np = gt_masks.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt_masks)

    if isinstance(pred_masks, torch.Tensor):
        pr_np = pred_masks.detach().cpu().numpy()
    else:
        pr_np = np.asarray(pred_masks)

    # ------------------------------------------------------
    # 1. AUTO-UNFLATTEN IF NEEDED
    # ------------------------------------------------------
    if gt_np.ndim == 2:
        N, HW = gt_np.shape

        # Infer H, W from HW (assumes square input, e.g. 256x256)
        side = int(np.sqrt(HW))
        if side * side != HW:
            raise ValueError(
                f"Cannot infer 2D mask shape from flattened size {HW}."
            )

        H = W = side

        gt_np = gt_np.reshape(N, H, W)
        pr_np = pr_np.reshape(N, H, W)

    # ------------------------------------------------------
    # Now shapes are guaranteed (N,H,W) or (N,1,H,W)
    # ------------------------------------------------------
    N = gt_np.shape[0]

    BIoU_list, HD95_list, MSD_list = [], [], []

    for i in range(N):
        gt_i = gt_np[i]
        pr_i = pr_np[i]

        # Remove channel dim (N,1,H,W)
        if gt_i.ndim == 3 and gt_i.shape[0] == 1:
            gt_i = gt_i[0]
        if pr_i.ndim == 3 and pr_i.shape[0] == 1:
            pr_i = pr_i[0]

        m = compute_boundary_metrics_single(gt_i, pr_i)

        # Skip invalid cases where any boundary metric is NaN
        if any(
            (v is None) or np.isnan(v)
            for v in (m["BIoU"], m["HD95"], m["MSD"])
        ):
            continue

        BIoU_list.append(m["BIoU"])
        HD95_list.append(m["HD95"])
        MSD_list.append(m["MSD"])

    # ------------------------------------------------------
    # All invalid → empty lists
    # ------------------------------------------------------
    if len(BIoU_list) == 0:
        return {
            "BIoU": [],
            "HD95": [],
            "MSD": [],
        }

    return {
        "BIoU": BIoU_list,
        "HD95": HD95_list,
        "MSD": MSD_list,
    }



def precision_recall_at_threshold(gt_masks, pred_masks, IOU_threshold=0.5, eps=1e-8):
    """Compute precision and recall for segmentation masks at a given IoU threshold."""
    if gt_masks.ndim == 3:
        gt_masks = gt_masks.flatten(1)
    if pred_masks.ndim == 3:
        pred_masks = pred_masks.flatten(1)

    G, P = gt_masks.shape[0], pred_masks.shape[0]

    if G == 0 and P == 0:
        return 1.0, 1.0
    if G == 0:
        return 0.0, 0.0
    if P == 0:
        return 0.0, 0.0

    intersection = torch.mm(pred_masks.float(), gt_masks.float().t())
    union = pred_masks.sum(1, keepdim=True) + gt_masks.sum(1).unsqueeze(0) - intersection
    ious = intersection / (union + eps)

    pred_matched = torch.zeros(P, dtype=torch.bool)
    gt_matched = torch.zeros(G, dtype=torch.bool)

    best_ious, _ = ious.max(dim=1)
    sorted_idx = torch.argsort(best_ious, descending=True)

    for i in sorted_idx:
        gt_idx = torch.argmax(ious[i])
        if ious[i, gt_idx] >= IOU_threshold and not gt_matched[gt_idx]:
            pred_matched[i] = True
            gt_matched[gt_idx] = True

    tp = pred_matched.sum().float()
    fp = (~pred_matched).sum().float()
    fn = (~gt_matched).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return precision.item(), recall.item()

'''
def precision_recall_at_threshold(gt_masks, pred_masks, IOU_threshold=0.0, eps=1e-8):
    """
    Compute precision and recall for segmentation masks at a given IoU threshold,
    using precomputed IoUs from mask_iou function.

    Args:
        gt_masks (torch.BoolTensor): Ground truth masks, shape [N, H*W]
        pred_masks (torch.BoolTensor): Predicted masks, shape [N, H*W]
        IOU_threshold (float): IoU threshold to count a prediction as true positive
        eps (float): Small epsilon to avoid division by zero

    Returns:
        precision (float), recall (float)
    """
    assert gt_masks.shape == pred_masks.shape, "GT and prediction masks must have the same shape"

    # Use precomputed IoUs
    ious = mask_iou(pred_masks, gt_masks)  # shape [N,]

    pred_nonempty = pred_masks.sum(dim=1) > 0
    gt_nonempty = gt_masks.sum(dim=1) > 0

    # True positives: prediction and GT exist, IoU >= threshold
    tp = torch.logical_and(ious > IOU_threshold, gt_nonempty).sum().float()

    # False negatives: GT exists but prediction IoU < threshold or prediction empty
    fn = torch.logical_and(gt_nonempty, ious <= IOU_threshold ).sum().float()

    # False positives: prediction exists but GT does NOT exist
    fp = torch.logical_and(pred_nonempty, ~gt_nonempty).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return precision.item(), recall.item()
'''

def compute_mPR(gt_masks, pred_masks, device, thresholds:list = [50, 95]):
    if thresholds[1] > 95:
        thresholds[1] = 95

    if thresholds[0] < 0:
        thresholds[0] = 0

    iou_thresholds = torch.arange(thresholds[0]/100, thresholds[1]/100 + 1e-6, 0.05, device=device)
    APs, ARs = [], []
    for thr in iou_thresholds:
        p, r = precision_recall_at_threshold(gt_masks, pred_masks, IOU_threshold = thr.item())
        APs.append(p)
        ARs.append(r)

    return float(torch.tensor(APs, device=device).mean()), float(torch.tensor(ARs, device=device).mean())

def get_best_threshold(y_true, y_score, thresholds):
    best_iou = 0
    best_threshold = 0.5
    for t in thresholds:
        y_pred = (y_score >= t).bool()
        y_true = y_true.bool()
        inter = torch.logical_and(y_true, y_pred).sum().float()
        union = torch.logical_or(y_true, y_pred).sum().float()
        iou = inter/union
        if iou > best_iou:
            best_iou = iou
            best_threshold = t.item()

    return best_iou, best_threshold

def compute_coverage(y_true, y_pred):

    TP = torch.logical_and(y_true, y_pred).sum(dim=1).float()
    PredP = y_pred.sum(dim=1).float()
    LabelP = y_true.sum(dim=1).float()
    haveP = PredP>0
    haveGT = LabelP>0

    miss = torch.logical_and(y_true, ~y_pred).sum(dim=1).float()
    miss[haveGT] = miss[haveGT] / (LabelP[haveGT] + eps)
    overflow = torch.logical_and(~y_true, y_pred).sum(dim=1).float() * haveGT
    overflow[haveGT] = overflow[haveGT] / (LabelP[haveGT] + eps)

    precisions = torch.zeros_like(TP)
    recalls = torch.zeros_like(TP)

    # Compute only for non-empty images
    precisions[haveP] = TP[haveP] / (PredP[haveP] + eps)
    recalls[haveGT] = TP[haveGT] / (LabelP[haveGT] + eps)

    # Mean over valid images
    mean_precision = precisions[haveP].mean().item() if haveP.sum() > 0 else 0.0
    mean_recall = recalls[haveGT].mean().item() if haveGT.sum() > 0 else 0.0
    mean_miss = miss[haveGT].mean().item() if haveGT.sum() > 0 else 0.0
    mean_overflow = overflow[haveGT].mean().item() if haveGT.sum() > 0 else 0.0

    return mean_precision, mean_recall, mean_miss, mean_overflow

def categorize_size_by_bbox(mask):
    """
    Categorize mask size based on bounding box area.
    mask: 2D tensor (H,W), binary
    Returns: 'small', 'medium', or 'large'
    """
    mask_bool = mask.bool()

    if mask_bool.sum() == 0:
        return 'empty'  # fallback for empty mask

    if mask_bool.size()[0] == 1:
        mask_bool = mask_bool.view((mask_bool.size()[1], mask_bool.size()[2]))

    # Find bounding box
    rows = torch.any(mask_bool, dim=1).nonzero(as_tuple=False).squeeze()
    cols = torch.any(mask_bool, dim=0).nonzero(as_tuple=False).squeeze()

    # If single-pixel object, nonzero returns 0D tensor, handle that
    if rows.ndim == 0:
        min_row, max_row = rows.item(), rows.item()
    else:
        min_row, max_row = rows[0].item(), rows[-1].item()

    if cols.ndim == 0:
        min_col, max_col = cols.item(), cols.item()
    else:
        min_col, max_col = cols[0].item(), cols[-1].item()

    bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)

    # Categorize based on COCO-style thresholds
    if bbox_area < 32**2:
        return 'small'
    elif bbox_area < 96**2:
        return 'medium'
    else:
        return 'large'

def compute_sml(gt_masks, pred_masks):
    # -----------------------------
    # Compute size-based APs/APm/APl & ARs/ARm/ARl
    # -----------------------------
    size_categories = defaultdict(lambda: defaultdict(lambda: 0))

    # For each image
    N = gt_masks.size()[0]
    for i in range(N):
        gt_mask = gt_masks[i]
        pred_mask = pred_masks[i]
        size = categorize_size_by_bbox(gt_mask)
        if size == 'empty':
            continue

        size_categories[size]['TP'] += float(torch.logical_and(pred_mask, gt_mask).sum())
        size_categories[size]['FP'] += float(torch.logical_and(pred_mask, ~gt_mask).sum())
        size_categories[size]['FN'] += float(torch.logical_and(~pred_mask, gt_mask).sum())

    # Aggregate size metrics
    def mean_prec_rec(metrics):
        TP = metrics['TP']
        FP = metrics['FP']
        FN = metrics['FN']
        precision = TP / (TP + FP + eps) if (TP + FP) > 0 else 0.0
        recall   = TP / (TP + FN + eps) if (TP + FN) > 0 else 0.0
        return precision, recall

    APs, ARs = mean_prec_rec(size_categories['small'])
    APm, ARm = mean_prec_rec(size_categories['medium'])
    APl, ARl = mean_prec_rec(size_categories['large'])

    return {
        'APs': APs,
        'APm': APm,
        'APl': APl,
        'ARs': ARs,
        'ARm': ARm,
        'ARl': ARl,
    }

def compute_segmentation_metrics(gt_masks, pred_masks, thresholds=None):
    """
    Pixel-level segmentation metrics (GPU-only).
    gt_masks: torch tensor (N,H,W), binary ground truth
    pred_masks: torch tensor (N,H,W), probabilities in [0,1]
    """
    device = gt_masks.device
    N = gt_masks.shape[0]

    # Best threshold by IoU
    if thresholds is None:
        thresholds = torch.round(torch.linspace(0, 1.0, 51, device=device), decimals = 2)

    best_iou, best_threshold = get_best_threshold(gt_masks, pred_masks, thresholds)
    # convert probability mask to segmentation mask
    pred_masks = (pred_masks > best_threshold).bool()
    APR = compute_sml(gt_masks.bool(), pred_masks)
    gt_masks = gt_masks.view(N, -1).bool()
    pred_masks = pred_masks.view(N, -1).bool()
    # Fixed thresholds
    AP50, AR50 = precision_recall_at_threshold(gt_masks, pred_masks, IOU_threshold = 0.5)
    AP30, AR30 = precision_recall_at_threshold(gt_masks, pred_masks, IOU_threshold = 0.3)
    AP, AR = precision_recall_at_threshold(gt_masks, pred_masks, IOU_threshold = 0)

    mAP, mAR = compute_mPR(gt_masks, pred_masks, device)

    precision, recall, miss, overflow = compute_coverage(gt_masks, pred_masks)

    result = {
        'best-threshold': best_threshold,
        'IoU': best_iou,
        'Dice-Coeff': 2 * best_iou/(best_iou + 1),
        'mAP': mAP,
        #'mAR': mAR,
        'AP50': AP50,
        #'AR50': AR50,
        'AP30': AP30,
        #'AR30': AR30,
        'AP': AP,
        #'AR': AR,
        'PixRecall': recall,
        'PixPrecision': precision,
        'miss': miss,
        'overflow': overflow
    }

    boundary = compute_boundary(gt_masks, pred_masks)
    for key in boundary:
        result[key] = boundary[key]

    # APR = compute_sml(gt_masks, pred_masks)
    for key in APR:
        result[key] = APR[key]

    return result

def export_IOU(gt_masks, pred_masks, thresholds=None):
    device = gt_masks.device
    N = gt_masks.shape[0]

    pred_masks = (pred_masks > thresholds).bool()
    iou = mask_iou(gt_masks.bool(), pred_masks)

    return iou

def export_miss(gt_masks, pred_masks, thresholds=None):
    device = gt_masks.device
    N = gt_masks.shape[0]

    pred_masks = (pred_masks > thresholds).bool()

    LabelP = gt_masks.bool().flatten(1).sum(dim=1).float()
    haveGT = LabelP>0

    miss = torch.logical_and(gt_masks.bool(), ~pred_masks).flatten(1).sum(dim=1).float()
    miss[haveGT] = miss[haveGT] / (LabelP[haveGT] + eps)
    return miss

def export_overflow(gt_masks, pred_masks, thresholds=None):
    device = gt_masks.device
    N = gt_masks.shape[0]

    pred_masks = (pred_masks > thresholds).bool()

    LabelP = gt_masks.bool().flatten(1).sum(dim=1).float()
    haveGT = LabelP>0

    overflow = torch.logical_and(~gt_masks.bool(), pred_masks).flatten(1).sum(dim=1).float() * haveGT
    overflow[haveGT] = overflow[haveGT] / (LabelP[haveGT] + eps)

    return overflow

# ======================================================
# Example usage
# ======================================================

class imageDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, cv):
        """
        initialize DatasetWrapper
        """
        super(imageDataset, self).__init__()
        self.f_name = []
        mbs = [os.path.join(cv, mb) for mb in os.listdir(cv) if os.path.isdir(os.path.join(cv, mb))]
        for mb in tqdm(mbs, desc = 'load data minibatch', leave = 0):
            imgs = [img for img in os.listdir(mb) if img.count('_') == 1]
            for img in tqdm(imgs, desc = 'load data image', leave = 0):
                files = [os.path.join(mb, file) for file in os.listdir(mb) if img.replace('.png','_') in file and ('label' in file or 'prop' in file)]
                files.sort()
                self.f_name.append(files)
        self.f_name.sort()
        with open(os.path.join(cv, 'filelist.txt'), "w") as f:
            for item in self.f_name:
                f.write(f"{item}\n")
        self.transformer = transforms.ToTensor()


    def __len__(self):
        # return size of dataset
        return len(self.f_name)


    def __getitem__(self, idx):

        try:
            mask_name, pred_mask = self.f_name[idx]
        except:
            print(self.f_name, idx, self.f_name[idx])
        mask = self.transformer(Image.open(mask_name))
        prop = self.transformer(Image.open(pred_mask))

        return mask, prop

def save_IOU_miss_to_txt(path, threshold=[0.5]*4):
    if os.path.exists(os.path.join(path,'iou.txt')) and os.path.exists(os.path.join(path,'miss.txt')):
        return

    cv_iters = [os.path.join(path, cv) for cv in os.listdir(path) if os.path.isdir(os.path.join(path, cv))]
    cv_iters.sort()
    ious, misses = [], []

    for i, cv in enumerate(tqdm(cv_iters, desc = path + ' CV', leave = 0)):
        gt_masks = torch.Tensor([])
        prob_masks = torch.Tensor([])
        dataloader = DataLoader(imageDataset(cv),
                        batch_size=20,
                        shuffle=True,
                        num_workers=6,
                        pin_memory=torch.cuda.is_available())

        for mask, prop in tqdm(dataloader, desc = 'make data: '+cv, leave = 0):
            gt_masks = torch.cat((gt_masks, mask), dim=0)
            prob_masks = torch.cat((prob_masks, prop), dim=0)

        # Step2: Object-level metrics (CPU)
        iou = export_IOU(gt_masks, prob_masks, threshold[i])
        miss = export_miss(gt_masks, prob_masks, threshold[i])
        ious.append(iou)
        misses.append(miss)

    ious = np.array(ious)
    np.savetxt(os.path.join(path,'iou.txt'), ious, delimiter=',')
    misses = np.array(misses)
    np.savetxt(os.path.join(path,'miss.txt'), misses, delimiter=',')

def save_matric_from_path(path, read_from_history = True):
    if os.path.exists(os.path.join(path, 'matrix.json')) and read_from_history:
        with open(os.path.join(path, 'matrix.json'), 'r') as file:
            metrics = json.load(file)
        save_IOU_miss_to_txt(path, metrics['best-threshold'])
        return metrics

    if os.path.isdir(os.path.join(path, '__pycache__')):
        shutil.rmtree(os.path.join(path, '__pycache__'))

    matrices = defaultdict(list)
    cv_iters = [os.path.join(path, cv) for cv in os.listdir(path) if os.path.isdir(os.path.join(path, cv))]
    cv_iters.sort()
    ious, misses = [], []
    BIOU, HD95, MSD = [], [], []

    for cv in tqdm(cv_iters, desc = path+' CV', leave = 0):
        gt_masks = torch.Tensor([])
        prob_masks = torch.Tensor([])
        dataloader = DataLoader(imageDataset(cv),
                        batch_size=20,
                        shuffle=False,
                        num_workers=6,
                        pin_memory=torch.cuda.is_available())

        for mask, prop in tqdm(dataloader, desc = 'make data: '+cv, leave = 0):
            gt_masks = torch.cat((gt_masks, mask), dim=0)
            prob_masks = torch.cat((prob_masks, prop), dim=0)

        # Step2: Object-level metrics (CPU)
        metrics = compute_segmentation_metrics(gt_masks, prob_masks)

        miss = export_miss(gt_masks, prob_masks, float(metrics['best-threshold']))
        misses.append(miss)
        del miss

        iou = export_IOU(gt_masks, prob_masks, float(metrics['best-threshold']))
        ious.append(iou)
        del iou

        BIOU = BIOU + metrics['BIoU']
        HD95 = HD95 + metrics['HD95']
        MSD = MSD + metrics['MSD']

        for key in metrics:
            matrices[key].append(float(np.mean([metrics[key]])))

    ious = np.array(ious)
    np.savetxt(os.path.join(path,'iou.txt'), ious, delimiter=',')
    misses = np.array(misses)
    np.savetxt(os.path.join(path,'miss.txt'), misses, delimiter=',')
    BIOU = np.array(BIOU)
    np.savetxt(os.path.join(path,'BIoU.txt'), BIOU, delimiter=',')
    HD95 = np.array(HD95)
    np.savetxt(os.path.join(path,'HD95.txt'), HD95, delimiter=',')
    MSD = np.array(MSD)
    np.savetxt(os.path.join(path,'MSD.txt'), MSD, delimiter=',')

    with open(os.path.join(path, 'matrix.json'), 'w') as file:
        json.dump(matrices, file, indent=4)

    return matrices


if __name__ == "__main__":
    path = '.'
    matrices = save_matric_from_path(path, False)
