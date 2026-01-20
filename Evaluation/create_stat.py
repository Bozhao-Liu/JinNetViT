import os
import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

def generate_pred_masks(root, read_from_history = True):
	# ------------------------------------------------------------
	# 1. Load best thresholds for each CV from matrix.json
	# ------------------------------------------------------------
	matrix_path = os.path.join(root, "matrix.json")
	with open(matrix_path, "r") as f:
		metrics = json.load(f)

	best_thresholds = metrics["best-threshold"]   # e.g. [0.47, 0.51, 0.49, 0.52]
	#print(best_thresholds)
	# ------------------------------------------------------------
	# 2. Locate CV folders (sorted)
	# ------------------------------------------------------------
	cv_folders = sorted([
		os.path.join(root, cv) 
		for cv in os.listdir(root)
		if os.path.isdir(os.path.join(root, cv)) and '_' in cv
	])

	tfs = transforms.ToTensor()

	# ------------------------------------------------------------
	# 3. Process each CV with its own best-threshold
	# ------------------------------------------------------------
	for cv_idx, cv_path in enumerate(tqdm(cv_folders, desc = 'Create prediction mask: CV',leave = 0)):
		thr = best_thresholds[cv_idx]
		#print(f"[CV {cv_idx+1}] Using best-threshold = {thr}")

		# minibatch folders
		mb_folders = [
			os.path.join(cv_path, mb)
			for mb in os.listdir(cv_path)
			if os.path.isdir(os.path.join(cv_path, mb))
		]
		for mb in tqdm(mb_folders, desc = 'mb',leave = 0):
			files = os.listdir(mb)

			# find all base images (exactly one underscore)
			base_imgs = [f for f in files if f.count("_") == 1 and f.endswith(".png")]
			for base in tqdm(base_imgs, desc = 'img',leave = 0):
				base_no_ext = base.replace(".png", "")

				save_name = base_no_ext + "_pred.png"
				if os.path.exists(os.path.join(mb, save_name)) and read_from_history:
					continue
				
				# prediction probability file
				pred_file = None
				for f in files:
					if f.startswith(base_no_ext + "_") and "prop" in f:
						pred_file = f
						break

				if pred_file is None:
					print(f"   No prop file found for {base}")
					continue

				pred_path = os.path.join(mb, pred_file)

				# ------------------------------------------------------------
				# 4. Load probability mask → threshold → binary mask
				# ------------------------------------------------------------
				prop = tfs(Image.open(pred_path))  # (1,H,W)
				prop = prop.squeeze(0)			 # (H,W)

				bin_mask = (prop > thr).float()

				# ------------------------------------------------------------
				# 5. Save output as <base>_pred.png
				# ------------------------------------------------------------
				out = (bin_mask * 255).byte().cpu().numpy()
				out_img = Image.fromarray(out).convert("L")

				out_img.save(os.path.join(mb, save_name))

				#print(f"   Saved {save_name}")


# ------------------------------------------------------
# Utilities copied from your getmatrix.py
# ------------------------------------------------------
eps = 1e-8

def mask_iou(mask1, mask2):
	inter = torch.logical_and(mask1, mask2).sum().float()
	union = torch.logical_or(mask1, mask2).sum().float()
	return (inter / (union + eps)).item()

def compute_dice(iou):
	return 2*iou / (1 + iou + eps)

def compute_miss(gt, pred):
	gt_pixels = gt.sum().float()
	if gt_pixels == 0:
		return 0.0
	miss = torch.logical_and(gt, ~pred).sum().float()
	return (miss / (gt_pixels + eps)).item()

def compute_overflow(gt, pred):
	gt_pixels = gt.sum().float()
	if gt_pixels == 0:
		return 0.0
	overflow = torch.logical_and(~gt, pred).sum().float()
	return (overflow / (gt_pixels + eps)).item()

# -------- boundary metric helpers --------
from scipy.ndimage import binary_erosion, distance_transform_edt

def extract_boundary(mask):
	structure = np.ones((3,3), bool)
	eroded = binary_erosion(mask, structure=structure, iterations=1, border_value=0)
	return mask ^ eroded

def surface_distance(gt, pr):
	if not gt.any() and not pr.any():
		return np.array([0.0])
	if gt.any() and not pr.any():
		return np.array([np.nan])
	if pr.any() and not gt.any():
		return np.array([np.nan])

	bg_pr = distance_transform_edt(~pr)
	bg_gt = distance_transform_edt(~gt)

	surf_gt = extract_boundary(gt)
	surf_pr = extract_boundary(pr)

	if not surf_gt.any() or not surf_pr.any():
		return np.array([np.nan])

	d1 = bg_pr[surf_gt]
	d2 = bg_gt[surf_pr]

	return np.concatenate([d1, d2])

def compute_boundary_metrics(gt_mask, pred_mask):
	gt = gt_mask.cpu().numpy().astype(bool)
	pr = pred_mask.cpu().numpy().astype(bool)

	gt_b = extract_boundary(gt)
	pr_b = extract_boundary(pr)

	inter = np.logical_and(gt_b, pr_b).sum()
	union = np.logical_or(gt_b, pr_b).sum()
	biou = 1.0 if union == 0 else inter / (union + eps)

	d = surface_distance(gt, pr)
	d_clean = d[~np.isnan(d)]
	if len(d_clean) == 0:
		return biou, np.nan, np.nan
	hd95 = float(np.percentile(d_clean, 95))
	msd = float(np.mean(d_clean))

	return biou, hd95, msd

# ------------------------------------------------------
# object size
# ------------------------------------------------------
def categorize_size_by_bbox(mask):
	# mask: torch.bool 2D
	if mask.sum() == 0:
		return "none"

	H, W = mask.shape
	rows = torch.any(mask, dim=1).nonzero(as_tuple=False).squeeze()
	cols = torch.any(mask, dim=0).nonzero(as_tuple=False).squeeze()

	if rows.ndim == 0:
		min_row = max_row = rows.item()
	else:
		min_row, max_row = rows[0].item(), rows[-1].item()

	if cols.ndim == 0:
		min_col = max_col = cols.item()
	else:
		min_col, max_col = cols[0].item(), cols[-1].item()

	area = (max_row - min_row + 1) * (max_col - min_col + 1)

	if area < 32**2:
		return "s"
	elif area < 96**2:
		return "m"
	else:
		return "l"


# ------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------

def generate_stat_file(root, read_from_history = True):
	tfs = transforms.ToTensor()
	out_path = os.path.join(root, "stat.txt")
	if read_from_history and os.path.exists(out_path):
		return
	# ------------------------------------------------------
	# 1. Collect all output rows in memory first
	# ------------------------------------------------------
	rows = []
	header = (
		"filepath,IoU,miss,Dice,BIoU,HD95,MSD,pred@IoU50,"
		"PixRecall,PixPrecision,overflow,objectsize"
	)
	rows.append(header)

	# locate CV folders
	cv_folders = sorted([
		os.path.join(root, cv)
		for cv in os.listdir(root)
		if os.path.isdir(os.path.join(root, cv)) and cv.lower().startswith("cv")
	])

	for cv in tqdm(cv_folders, desc = 'Create stat file: CV',leave = 0):
		mb_folders = [
			os.path.join(cv, mb)
			for mb in os.listdir(cv)
			if os.path.isdir(os.path.join(cv, mb))
		]

		for mb in tqdm(mb_folders, desc = 'mb',leave = 0):
			files = os.listdir(mb)
			base_imgs = [f for f in files if f.count("_") == 1 and f.endswith(".png")]

			for base in tqdm(base_imgs, desc = 'img',leave = 0):
				base_no_ext = base.replace(".png", "")

				gt_file   = base_no_ext + "_label.png"
				pred_file = base_no_ext + "_pred.png"

				if gt_file not in files or pred_file not in files:
					continue

				gt_path   = os.path.join(mb, gt_file)
				pred_path = os.path.join(mb, pred_file)

				# load GT and pred masks
				gt = tfs(Image.open(gt_path)).squeeze(0).bool()
				pr = tfs(Image.open(pred_path)).squeeze(0).bool()

				# -------------------------------------
				# Compute metrics
				# -------------------------------------
				iou = mask_iou(gt, pr)
				dice = compute_dice(iou)
				miss = compute_miss(gt, pr)
				overflow = compute_overflow(gt, pr)

				biou, hd95, msd = compute_boundary_metrics(gt, pr)

				# pixel precision/recall
				tp = torch.logical_and(gt, pr).sum().float()
				predP = pr.sum().float()
				gtP   = gt.sum().float()

				pix_precision = (tp / (predP + eps)).item() if predP > 0 else 0.0
				pix_recall	= (tp / (gtP + eps)).item()   if gtP > 0 else 0.0

				pred_at_50 = 1 if iou >= 0.5 else 0
				objsize = categorize_size_by_bbox(gt)

				# -------------------------------------
				# Create CSV row
				# -------------------------------------
				row = (
					f"{pred_path},{iou:.6f},{miss:.6f},{dice:.6f},"
					f"{biou:.6f},{hd95:.6f},{msd:.6f},{pred_at_50},"
					f"{pix_recall:.6f},{pix_precision:.6f},{overflow:.6f},{objsize}"
				)
				rows.append(row)

	# ------------------------------------------------------
	# 2. Write all rows at once (fast & safe)
	# ------------------------------------------------------
	with open(out_path, "w") as fw:
		fw.write("\n".join(rows))


if __name__ == "__main__":
	path = '.'

	generate_pred_masks(path)
	generate_stat_file(path)
