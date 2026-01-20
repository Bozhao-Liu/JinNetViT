import os
from PIL import Image
import numpy as np
import re

def compare_all_experiments(parent_dir, check_content=True):
    """
    Compare image_<something>.png consistency across ALL (network, loss) combinations.

    Expected structure:
      parent_dir/
        ├── Network1/
        │   ├── Loss1/
        │   │   ├── Fold1/
        │   │   │   ├── Batch1/
        │   │   │   │   ├── image_123.png
        │   │   │   │   ├── ...
        │   ├── Loss2/
        ├── Network2/
        │   ├── Loss1/
        │   ├── Loss2/
    """

    def get_subdirs(path):
        return sorted(
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        )

    def get_image_files(folder):
        """
        Return only base images like 'img_<number>.png'
        (exclude any *_label.png, *_pred.png, *_prop.png, etc.)
        """
        pattern = re.compile(r"^img_\d+\.png$", re.IGNORECASE)
        keep = set()
        for f in os.listdir(folder):
            if pattern.match(f):
                keep.add(f)
        return keep

    # --- 1. Detect networks ---
    network_dirs = sorted(
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    )
    if len(network_dirs) < 1:
        print("❌ No network folders found.")
        return

    network_names = [os.path.basename(d) for d in network_dirs]
    print("✅ Found networks:")
    for n in network_names:
        print("   ", n)

    # --- 2. Check that all networks have the same loss folders ---
    loss_sets = []
    for net_dir in network_dirs:
        losses = sorted(
            d for d in os.listdir(net_dir)
            if os.path.isdir(os.path.join(net_dir, d))
        )
        loss_sets.append(losses)

    ref_losses = set(loss_sets[0])
    all_loss_match = all(set(ls) == ref_losses for ls in loss_sets)

    if not all_loss_match:
        print("\n❌ Loss folder mismatch among networks!")
        for i, ls in enumerate(loss_sets):
            missing = ref_losses - set(ls)
            extra = set(ls) - ref_losses
            net_name = network_names[i]
            if missing:
                print(f"  Network '{net_name}' missing losses: {missing}")
            if extra:
                print(f"  Network '{net_name}' has extra losses: {extra}")
        return

    print("\n✅ All networks share the same loss folders:", sorted(ref_losses))

    # --- 3. Build list of all experiments (network, loss, root) ---
    experiments = []
    for net_dir, net_name in zip(network_dirs, network_names):
        for loss_name in sorted(ref_losses):
            root = os.path.join(net_dir, loss_name)
            experiments.append({
                "net": net_name,
                "loss": loss_name,
                "root": root,
            })

    print("\n✅ Experiments to compare:")
    for exp in experiments:
        print(f"   {exp['net']} / {exp['loss']}")

    if not experiments:
        print("❌ No experiments found.")
        return

    # Use first experiment as reference
    ref_exp = experiments[0]
    ref_root = ref_exp["root"]
    print(f"\n📌 Using reference experiment: {ref_exp['net']} / {ref_exp['loss']}")

    # --- 4. Compare folds across all experiments ---
    ref_folds = set(get_subdirs(ref_root))
    if not ref_folds:
        print("❌ No folds found in reference experiment.")
        return

    print("\n✅ Folds in reference:", sorted(ref_folds))

    for exp in experiments[1:]:
        folds = set(get_subdirs(exp["root"]))
        if folds != ref_folds:
            missing = ref_folds - folds
            extra = folds - ref_folds
            print(f"\n❌ Fold mismatch in {exp['net']} / {exp['loss']}")
            if missing:
                print(f"   Missing folds: {missing}")
            if extra:
                print(f"   Extra folds:   {extra}")

    # --- 5. For each fold, compare minibatches and images ---
    for fold in sorted(ref_folds):
        print(f"\n==============================")
        print(f"🔎 Checking fold: {fold}")
        print(f"==============================")

        ref_fold_root = os.path.join(ref_root, fold)
        ref_batches = set(get_subdirs(ref_fold_root))
        if not ref_batches:
            print(f"  ⚠️ No minibatches found in reference for fold '{fold}'")
            continue

        # 5.1: Check batches across all experiments
        for exp in experiments[1:]:
            exp_fold_root = os.path.join(exp["root"], fold)
            batches = set(get_subdirs(exp_fold_root))
            if batches != ref_batches:
                missing = ref_batches - batches
                extra = batches - ref_batches
                print(f"  ❌ Minibatch mismatch in {exp['net']} / {exp['loss']} for fold '{fold}'")
                if missing:
                    print(f"     Missing minibatches: {missing}")
                if extra:
                    print(f"     Extra minibatches:   {extra}")

        # 5.2: For each batch, compare image_* files
        for batch in sorted(ref_batches):
            print(f"\n  → Checking minibatch: {batch}")
            ref_batch_root = os.path.join(ref_fold_root, batch)

            # images in reference
            ref_images = get_image_files(ref_batch_root)

            if not ref_images:
                print(f"    ⚠️ No image_*.png in reference minibatch: {ref_batch_root}")
                # Optional: uncomment to debug
                # print("    Contents:", os.listdir(ref_batch_root))
                continue

            # Compare filenames in all other experiments
            for exp in experiments[1:]:
                exp_batch_root = os.path.join(exp["root"], fold, batch)
                exp_images = get_image_files(exp_batch_root)

                if exp_images != ref_images:
                    missing = ref_images - exp_images
                    extra = exp_images - ref_images
                    print(f"    ❌ Filename mismatch in {exp['net']} / {exp['loss']}")
                    if missing:
                        print(f"       Missing images: {missing}")
                    if extra:
                        print(f"       Extra images:   {extra}")

            print("    ✅ image_*.png filenames match (where no errors printed above).")

            # 5.3: Strict pixel content check
            if check_content:
                for img_name in ref_images:
                    # Load reference image once
                    ref_img_path = os.path.join(ref_batch_root, img_name)
                    with Image.open(ref_img_path) as im_ref:
                        ref_arr = np.array(im_ref.convert("RGB"))

                    for exp in experiments[1:]:
                        exp_img_path = os.path.join(exp["root"], fold, batch, img_name)
                        with Image.open(exp_img_path) as im_exp:
                            exp_arr = np.array(im_exp.convert("RGB"))

                        if exp_arr.shape != ref_arr.shape:
                            print(
                                f"    ⚠️ Shape mismatch for {img_name} in fold '{fold}', minibatch '{batch}' "
                                f"between reference and {exp['net']} / {exp['loss']}"
                            )
                            continue

                        if not np.array_equal(exp_arr, ref_arr):
                            print(
                                f"    ❌ Pixel mismatch for {img_name} in fold '{fold}', "
                                f"minibatch '{batch}' for {exp['net']} / {exp['loss']}"
                            )

                print("    ✅ Content check complete for this minibatch (where no mismatches printed).")



if __name__ == "__main__":
    parent = "./Result/prediction"  # change this
    compare_all_experiments(parent, check_content=True)
