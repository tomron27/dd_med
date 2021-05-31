import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from config import DATA_DIR

train_root = [
    os.path.join(DATA_DIR, "train/LGG"),
    os.path.join(DATA_DIR, "train/HGG"),
]

dest_train = os.path.join(DATA_DIR, "train_split_proc_new/")


def dilute_and_save_image_and_mask(x, mask, dest, folder):
    """
    Skips slices without tumor and saves to .npz file
    """
    mask_filtered = []
    x_filtered = []
    start_idx = None
    for i in range(x.shape[0]):
        area = mask[i].sum()
        if area > 0:
            if start_idx is None:
                start_idx = i
            mask_filtered.append(mask[i])
            x_filtered.append(x[:, i])
    x_filtered = np.array(x_filtered, dtype=np.float64)
    x_filtered = np.transpose(x_filtered, (1, 0, 2, 3))
    mask_filtered = np.array(mask_filtered, dtype=np.uint8)

    # Save slices as npy 2d arrays
    os.makedirs(os.path.join(dest, folder), exist_ok=True)
    for i in range(x_filtered.shape[1]):
        area = mask_filtered[i].sum()
        img_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_y={area}.npz")
        mask_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_mask.npz")
        # img_fname = os.path.join(dest, folder + f"_slice={str(start_idx + i)}_y={area}.npy")
        # mask_fname = os.path.join(dest, folder + f"_slice={str(start_idx + i)}_mask.npy")
        try:
            np.savez_compressed(img_fname, data=x_filtered[:, i])
            np.savez_compressed(mask_fname, data=mask_filtered[i])
            # np.save(img_fname, x_filtered[:, i])
            # np.save(mask_fname, mask_filtered[i])
        except:
            print(f"Error occurred on {img_fname}, continuing")


def dilute_split_and_save_image_and_mask(x, mask, dest, folder, g_type):
    """
    Skips slices without tumor and saves to .npz file
    """
    try:
        mask_filtered = []
        x_filtered = []
        start_idx = None
        for i in range(x.shape[1]):
            area = mask[i].sum()
            if area > 1000:
                if start_idx is None:
                    start_idx = i
                mask_filtered.append(mask[i])
                x_filtered.append(x[:, i])
        x_filtered = np.array(x_filtered, dtype=np.float64)
        x = np.transpose(x_filtered, (1, 0, 2, 3))
        mask = np.array(mask_filtered, dtype=np.uint8)

        # Save slices as npy 2d arrays
        os.makedirs(os.path.join(dest, folder), exist_ok=True)
        for i in range(x.shape[1]):

            x_min, x_max = np.where(mask[i].sum(axis=0))[0].min(), np.where(mask[i].sum(axis=0))[0].max()
            # y_min, y_max = np.where(mask[i].sum(axis=1))[0].min(), np.where(mask[i].sum(axis=1))[0].max()

            # plt.imshow(x[2][i], cmap="gray")
            # plt.imshow(mask[i], cmap="jet", alpha=0.4)
            #
            # plt.axhline(y=y_min, color='r', linestyle='-')
            # plt.axhline(y=y_max, color='r', linestyle='-')
            # plt.axvline(x=x_min, color='b', linestyle='-')
            # plt.axvline(x=x_max, color='b', linestyle='-')
            # plt.show()

            x_right, x_left = np.zeros(shape=(x.shape[0], *x.shape[2:]), dtype=x.dtype), np.zeros(shape=(x.shape[0], *x.shape[2:]), dtype=x.dtype)
            mask_right, mask_left = np.zeros(shape=mask.shape[1:], dtype=mask.dtype), np.zeros(shape=mask.shape[1:], dtype=mask.dtype)

            left = 1
            if (x_min + x_max) / 2 < 120:   # Tumor is on the left side
                x_left[:, :, 0:x_max], x_right[:, :, x_max:240] = x[:, i, :, 0:x_max], x[:, i, :, x_max:240]
                mask_left[:, 0:x_max], mask_right[:, x_max:240] = mask[i, :, 0:x_max], mask[i, :, x_max:240]
            else:
                left = 0
                x_left[:, :, 0:x_min], x_right[:, :, x_min:240] = x[:, i, :, 0:x_min], x[:, i, :, x_min:240]
                mask_left[:, 0:x_min], mask_right[:, x_min:240] = mask[i, :, 0:x_min], mask[i, :, x_min:240]

            # y = 1 if g_type == "LGG" else 2
            # if left:
            #     left_img_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_side=left_y={y}.npz")
            #     right_img_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_side=right_y={0}.npz")
            #     left_mask_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_side=left_y={y}_mask.npz")
            #     right_mask_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_side=right_y={0}_mask.npz")
            # else:
            #     left_img_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_side=left_y={0}.npz")
            #     right_img_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_side=right_y={y}.npz")
            #     left_mask_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_side=left_y={0}_mask.npz")
            #     right_mask_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_side=right_y={y}_mask.npz")

            left_img_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_side=left_y={left}.npy")
            right_img_fname = os.path.join(dest, folder, folder + f"_slice={str(start_idx + i)}_side=right_y={1-left}.npy")
            left_mask_fname = os.path.join(dest, folder,
                                           folder + f"_slice={str(start_idx + i)}_side=left_y={left}_mask.npy")
            right_mask_fname = os.path.join(dest, folder,
                                            folder + f"_slice={str(start_idx + i)}_side=right_y={1-left}_mask.npy")
            with open(left_img_fname, 'wb') as f:
                np.save(f, x_left)
            with open(right_img_fname, 'wb') as f:
                np.save(f, x_right)
            with open(left_mask_fname, 'wb') as f:
                np.save(f, mask_left)
            with open(right_mask_fname, 'wb') as f:
                np.save(f, mask_right)

            # np.savez_compressed(left_img_fname, data=x_left)
            # np.savez_compressed(right_img_fname, data=x_right)
            # np.savez_compressed(left_mask_fname, data=mask_left)
            # np.savez_compressed(right_mask_fname, data=mask_right)
    except Exception as e:
        # raise e
        print(f"Error occurred on {folder}:", str(e))


if __name__ == "__main__":
    for train_folder in train_root:
        print(f"Probing folder '{train_folder}'")
        g_type = os.path.basename(train_folder)
        folders = os.listdir(train_folder)
        os.makedirs(dest_train, exist_ok=True)
        for i, folder in tqdm(enumerate(folders), total=len(folders)):
            if g_type == "HGG" and i in (156, 198):
                continue
            # print(g_type, i)
            t1_img = nib.load(os.path.join(train_folder, folder, folder + "_t1.nii.gz")).get_fdata()
            t1_img = np.rot90(t1_img, k=1, axes=(0, 2))
            t1ce_img = nib.load(os.path.join(train_folder, folder, folder + "_t1ce.nii.gz")).get_fdata()
            t1ce_img = np.rot90(t1ce_img, k=1, axes=(0, 2))
            t2_img = nib.load(os.path.join(train_folder, folder, folder + "_t2.nii.gz")).get_fdata()
            t2_img = np.rot90(t2_img, k=1, axes=(0, 2))
            flair_img = nib.load(os.path.join(train_folder, folder, folder + "_flair.nii.gz")).get_fdata()
            flair_img = np.rot90(flair_img, k=1, axes=(0, 2))

            mask = nib.load(os.path.join(train_folder, folder, folder + "_seg.nii.gz")).get_fdata()
            mask = np.rot90(mask, k=1, axes=(0, 2))

            mask = (mask > 0).astype(np.uint8)
            x = np.zeros(shape=(4, *t1_img.shape), dtype=np.float64)
            x[0] = t1_img
            x[1] = t1ce_img
            x[2] = t2_img
            x[3] = flair_img

            dilute_split_and_save_image_and_mask(x, mask, dest_train, folder, g_type)


    # t2_img = nib.load(os.path.join(sample_dir, "Brats18_TCIA08_469_1_t2.nii.gz")).get_fdata()
    # t2_img = np.rot90(t2_img, k=1, axes=(0, 2))
    # mask = nib.load(os.path.join(sample_dir, "Brats18_TCIA08_469_1_seg.nii.gz")).get_fdata()
    # mask = np.rot90(mask, k=1, axes=(0, 2))
    # idx = 10
    # t2_img_slice = t2_img[idx]
    # mask_slice = mask[idx]
    # mask_slice = (mask_slice > 0).astype(np.uint8)
    #
    # plt.imshow(t2_img_slice, cmap="gray")
    # plt.imshow(mask_slice, cmap="jet", alpha=0.5)
    # plt.show()
    # print(mask_slice.sum())
    # pass