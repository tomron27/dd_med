import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from torchvision.transforms import InterpolationMode
from kornia.augmentation import Normalize
from config import TrainConfig, PROJECT_DIR
from dataio.dataloader import probe_data_folder, BraTS18Binary
from models.unet import get_unet_encoder_classifier
from models.attention import SumPool


# Ignore pytorch warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def get_dataset():
    raw_params = TrainConfig()
    # Serialize transforms
    train_transforms = raw_params.config.pop("train_transforms")
    test_transforms = raw_params.config.pop("test_transforms")
    raw_params.config["train_transforms"] = {"transforms": train_transforms.__str__()}
    raw_params.config["test_transforms"] = {"test_transforms": test_transforms.__str__()}

    params = raw_params.config

    # Load dataset
    train_metadata, val_metadata, _ = probe_data_folder(params["data_path"],
                                                        train_frac=params["train_frac"],
                                                        bad_files=params["bad_files"],
                                                        subsample_frac=params["subsample_frac"])

    transforms = Compose([
        Normalize(mean=torch.tensor([149.56119, 165.83559, 166.13501, 112.61901]),
                  std=torch.tensor([636.8766, 653.8386, 759.8256, 718.83594])),
        Resize((256, 256), interpolation=InterpolationMode.NEAREST)])

    mask_transforms = Resize((256, 256), interpolation=InterpolationMode.NEAREST)

    # Dataset
    # Filter only positive classes
    val_metadata = [(image, mask) for (image, mask) in val_metadata if "y=1" in image]

    # val_metadata = val_metadata[:100]   #DEBUG
    val_dataset = BraTS18Binary(params["data_path"],
                                val_metadata,
                                transforms=transforms,
                                mask_transforms=mask_transforms,
                                prefetch_data=params["prefetch_data"],
                                get_mask=True,
                                shuffle=False)
    # Dataloader
    val_loader = DataLoader(dataset=val_dataset,
                            num_workers=1,
                            pin_memory=True,
                            batch_size=1,
                            shuffle=False)
    return val_dataset, val_loader


def get_base_model(model_path):
    base_dir, model_file = os.path.split(model_path)

    params = TrainConfig().config
    params["learnable_attn"] = False
    params["learnable_duals"] = False
    params["attn_kl"] = False
    params["weights"] = model_path = os.path.join(base_dir, model_file)

    # Cuda
    model = get_unet_encoder_classifier(**params)
    model = model.to(device)
    model.eval()

    return model


def save_images_and_masks():
    res = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        res.append((image[0].numpy(), target.item(), mask[0].numpy()))
    with open(f'images_and_masks.pkl', 'wb') as f:
        pickle.dump(res, f)


def get_deeplift_attr(model, block=4, downsample=True, normalize=True):
    attrs = []
    from captum.attr import DeepLift
    attributer = DeepLift(model, multiply_by_inputs=False)
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        attr = attributer.attribute(image, target=target[0])
        if normalize:
            attr -= attr.view(attr.shape[0], attr.shape[1], -1).min(dim=2)[0][:, :, None, None]
            attr /= attr.view(attr.shape[0], attr.shape[1], -1).max(dim=2)[0][:, :, None, None]
            attr = attr.sum(dim=1)
            attr = torch.softmax(attr.view(1, -1), dim=1).reshape(attr.shape)
        if downsample:
            if block == 4:
                attr = SumPool(8)(attr)
            elif block == 3:
                attr = SumPool(4)(attr)
            elif block == 2:
                attr = SumPool(2)(attr)
            elif block == 1:
                pass
            else:
                raise NotImplementedError
        attrs.append(attr.detach().cpu())
    return attrs


def get_gradcam_attr(model, block=4, normalize=True):
    attrs = []
    from captum.attr import LayerGradCam
    if block == 4:
        layer = model.encoder4
    elif block == 3:
        layer = model.encoder3
    elif block == 2:
        layer = model.encoder2
    elif block == 1:
        layer = model.encoder1
    else:
        raise NotImplementedError
    attributer = LayerGradCam(model, layer=layer)
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        attr = attributer.attribute(image, target=1 - target.item())  # Bug?
        if normalize:
            attr -= attr.view(attr.shape[0], attr.shape[1], -1).min(dim=2)[0][:, :, None, None]
            attr /= attr.view(attr.shape[0], attr.shape[1], -1).max(dim=2)[0][:, :, None, None]
            attr = attr.sum(dim=1)
            attr = torch.softmax(attr.view(1, -1), dim=1).reshape(attr.shape)
        attrs.append(attr.detach().cpu())
    return attrs


def get_lrp_attr(model, block=4, normalize=True, downsample=True):
    attrs = []
    from captum.attr import LRP
    attributer = LRP(model)
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        attr = attributer.attribute(image, target=target[0])
        if normalize:
            attr -= attr.view(attr.shape[0], attr.shape[1], -1).min(dim=2)[0][:, :, None, None]
            attr /= attr.view(attr.shape[0], attr.shape[1], -1).max(dim=2)[0][:, :, None, None]
            attr = attr.sum(dim=1)
            attr = torch.softmax(attr.view(1, -1), dim=1).reshape(attr.shape)
        if downsample:
            if block == 4:
                attr = SumPool(8)(attr)
            elif block == 3:
                attr = SumPool(4)(attr)
            elif block == 2:
                attr = SumPool(2)(attr)
            elif block == 1:
                pass
            else:
                raise NotImplementedError
        attrs.append(attr.detach().cpu())
    return attrs


def get_inxgrad_attr(model, block=4, normalize=True, downsample=True):
    attrs = []
    from captum.attr import InputXGradient
    attributer = InputXGradient(model)
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        attr = attributer.attribute(image, target=target[0])
        if normalize:
            attr -= attr.view(attr.shape[0], attr.shape[1], -1).min(dim=2)[0][:, :, None, None]
            attr /= attr.view(attr.shape[0], attr.shape[1], -1).max(dim=2)[0][:, :, None, None]
            attr = attr.sum(dim=1)
            attr = torch.softmax(attr.view(1, -1), dim=1).reshape(attr.shape)
        if downsample:
            if block == 4:
                attr = SumPool(8)(attr)
            elif block == 3:
                attr = SumPool(4)(attr)
            elif block == 2:
                attr = SumPool(2)(attr)
            elif block == 1:
                pass
            else:
                raise NotImplementedError
        attrs.append(attr.detach().cpu())
    return attrs


def get_dual_attr(model_path, block=4):
    base_dir, model_file = os.path.split(model_path)
    params = TrainConfig().config
    params["learnable_attn"] = True
    params["learnable_duals"] = True
    params["attn_kl"] = True
    params["weights"] = model_path = os.path.join(base_dir, model_file)
    model = get_unet_encoder_classifier(**params)
    model = model.to(device)
    model.eval()

    attns = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        outputs, attn, duals = model(image)
        attns.append(attn[block - 3].detach().cpu())
    return attns

def get_ap(mask, heatmap):
    mask = mask[0]
    heatmap = heatmap[0]
    if heatmap.shape != mask.shape:
        heatmap = resize_filter(torch.tensor(heatmap).unsqueeze(0)).squeeze(0) / (
                    (heatmap.shape[0] / mask.shape[0]) ** 2)
    heatmap = heatmap.numpy()
    return average_precision_score(mask.ravel(), heatmap.ravel())


def get_q_iou(mask, heatmap, q=0.975):
    if heatmap.shape != mask.shape:
        heatmap = resize_filter(torch.tensor(heatmap).unsqueeze(0)).squeeze(0) / (
                    (heatmap.shape[0] / mask.shape[0]) ** 2)
    heatmap = heatmap.numpy()
    masked_heatmap = (heatmap > np.quantile(heatmap, 1 - q))
    mask = mask.numpy().astype(np.bool)
    intersect = (mask * masked_heatmap).sum()
    union = (mask + masked_heatmap).sum()
    return intersect / union


def get_ap_stats(attrs):
    ap_list = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (_, mask) = sample
        ap = get_ap(mask, attrs[j])
        ap_list.append(ap)
    return sum(ap_list) / len(ap_list)


def get_q_iou_stats(attrs, q=0.95):
    qiou_list = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (_, mask) = sample
        qiou = get_q_iou(mask, attrs[j], q=1-q)
        qiou_list.append(qiou)
    return sum(qiou_list) / len(qiou_list)


if __name__ == "__main__":

    baseline_model_path = os.path.join(PROJECT_DIR, "models/unet_encoder_baseline_score=0.9717.pt")
    dual_model_path = os.path.join(PROJECT_DIR, "models/unet_encoder_dual_decomp_score=0.9736.pt")

    calc_attrs = True
    calc_overlap = True
    blocks = [3]
    resize_filter = Resize((256, 256), interpolation=InterpolationMode.NEAREST)

    # Base model and dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset, val_loader = get_dataset()
    base_model = get_base_model(baseline_model_path)
    
    os.makedirs("attrs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print("*** Baseline methods ***")
    for block in blocks:
        print(f"*** Block {block} ***")
        # Get attribution maps
        if calc_attrs:
            if not os.path.exists(f'attrs/gc_attrs_block={block}.pkl'):
                print(f"Calculating attributions: GracCAM block={block}")
                gc_attrs = get_gradcam_attr(base_model, block=block)
                with open(f'attrs/gc_attrs_block={block}.pkl', 'wb') as f:
                    pickle.dump(gc_attrs, f)
            if not os.path.exists(f'attrs/dl_attrs_block={block}.pkl'):
                print(f"Calculating attributions: DeepLIFT block={block}")
                dl_attrs = get_deeplift_attr(base_model, block=block)
                with open(f'attrs/dl_attrs_block={block}.pkl', 'wb') as f:
                    pickle.dump(dl_attrs, f)
            if not os.path.exists(f'attrs/lrp_attrs_block={block}.pkl'):
                print(f"Calculating attributions: LRP block={block}")
                lrp_attrs = get_lrp_attr(base_model, block=block)
                with open(f'attrs/lrp_attrs_block={block}.pkl', 'wb') as f:
                    pickle.dump(lrp_attrs, f)
            if not os.path.exists(f'attrs/dual_attrs_block={block}.pkl'):
                print(f"Calculating attributions: Dual block={block}")
                dual_attrs = get_dual_attr(dual_model_path, block=block)
                with open(f'attrs/dual_attrs_block={block}.pkl', 'wb') as f:
                    pickle.dump(dual_attrs, f)

        # Calculate overlap
        if calc_overlap:
            with open(f'attrs/gc_attrs_block={block}.pkl', 'rb') as f:
                gc_attrs = pickle.load(f)
            with open(f'attrs/dl_attrs_block={block}.pkl', 'rb') as f:
                dl_attrs = pickle.load(f)
            with open(f'attrs/lrp_attrs_block={block}.pkl', 'rb') as f:
                lrp_attrs = pickle.load(f)
            with open(f'attrs/dual_attrs_block={block}.pkl', 'rb') as f:
                dual_attrs = pickle.load(f)

            methods = {
                "GradCAM": gc_attrs,
                "DeepLIFT": dl_attrs,
                "LRP": lrp_attrs,
                "Dual": dual_attrs,
            }

            for name, attrs in methods.items():
                print(f"Calculating overlap for: , method={name}, block={block}")
                if not os.path.exists(f'results/method={name}_block={block}__qiou_q=0.90.pkl'):
                    stats = get_q_iou_stats(attrs, q=0.90)
                    with open(f'results/method={name}_block={block}__qiou_q=0.90.pkl', 'wb') as f:
                        pickle.dump(stats, f)
                if not os.path.exists(f'results/method={name}_block={block}__qiou_q=0.95.pkl'):
                    stats = get_q_iou_stats(attrs, q=0.95)
                    with open(f'results/method={name}_block={block}__qiou_q=0.95.pkl', 'wb') as f:
                        pickle.dump(stats, f)
                if not os.path.exists(f'results/method={name}_block={block}__qiou_q=0.975.pkl'):
                    stats = get_q_iou_stats(attrs, q=0.975)
                    with open(f'results/method={name}_block={block}__qiou_q=0.975.pkl', 'wb') as f:
                        pickle.dump(stats, f)
                if not os.path.exists(f'results/method={name}_block={block}__map.pkl'):
                    stats = get_ap_stats(attrs)
                    with open(f'results/method={name}_block={block}__map.pkl', 'wb') as f:
                        pickle.dump(stats, f)
    # Print results
    results = os.listdir("results")
    result_index = ["method", "block", "metric", "value"]
    result_df = pd.DataFrame(columns=result_index)
    for result in results:
        method, block, metric = re.search(r"method=([^\_]+)_block=([0-9]+)__(.+).pkl", result).groups()
        with open(os.path.join("results", result), 'rb') as f:
            value = pickle.load(f)
            result_df = result_df.append({"method": method, "block": block, "metric": metric, "value": value},
                                         ignore_index=True)
    result_df = result_df.pivot(index=['block', 'metric'], columns=['method'], values=['value']) * 100
    print(result_df)
