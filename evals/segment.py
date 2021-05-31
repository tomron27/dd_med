import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from torchvision.transforms import InterpolationMode
from kornia.augmentation import Normalize

from config import TrainConfig
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


def get_deeplift_attr(model, level=4, downsample=True, normalize=True):
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
            if level == 4:
                attr = SumPool(8)(attr)
            elif level == 3:
                attr = SumPool(4)(attr)
            elif level == 2:
                attr = SumPool(2)(attr)
            elif level == 1:
                pass
            else:
                raise NotImplementedError
        attrs.append(attr.detach().cpu())
    return attrs


def get_gradcam_attr(model, level=4, normalize=True):
    attrs = []
    from captum.attr import LayerGradCam
    if level == 4:
        layer = model.encoder4
    elif level == 3:
        layer = model.encoder3
    elif level == 2:
        layer = model.encoder2
    elif level == 1:
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


def get_lrp_attr(model, level=4, normalize=True, downsample=True):
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
            if level == 4:
                attr = SumPool(8)(attr)
            elif level == 3:
                attr = SumPool(4)(attr)
            elif level == 2:
                attr = SumPool(2)(attr)
            elif level == 1:
                pass
            else:
                raise NotImplementedError
        attrs.append(attr.detach().cpu())
    return attrs


def get_inxgrad_attr(model, level=4, normalize=True, downsample=True):
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
            if level == 4:
                attr = SumPool(8)(attr)
            elif level == 3:
                attr = SumPool(4)(attr)
            elif level == 2:
                attr = SumPool(2)(attr)
            elif level == 1:
                pass
            else:
                raise NotImplementedError
        attrs.append(attr.detach().cpu())
    return attrs


def get_dual_attr(model_path, level=4, config_file="params.p"):
    base_dir, model_file = os.path.split(model_path)

    config_handler = open(os.path.join(base_dir, config_file), 'rb')
    params = pickle.load(config_handler)
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
        attns.append(attn[level - 3].detach().cpu())
    return attns


def get_baseline_attr(model_path, level=4, config_file="params.p"):
    base_dir, model_file = os.path.split(model_path)

    config_handler = open(os.path.join(base_dir, config_file), 'rb')
    params = pickle.load(config_handler)
    params["learnable_attn"] = True
    params["learnable_duals"] = False
    params["attn_kl"] = False
    params["weights"] = model_path = os.path.join(base_dir, model_file)
    model = get_unet_encoder_classifier(**params)
    model = model.to(device)
    model.eval()

    attns = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (target, mask) = sample
        image = image.to(device)
        outputs, attn, _ = model(image)
        attns.append(attn[level - 3].detach().cpu())
    return attns


def get_ap(mask, heatmap):
    mask = mask[0]
    heatmap = heatmap[0]
    if heatmap.shape != mask.shape:
        heatmap = resize_filter(torch.tensor(heatmap).unsqueeze(0)).squeeze(0) / (
                    (heatmap.shape[0] / mask.shape[0]) ** 2)
    heatmap = heatmap.numpy()
    return average_precision_score(mask.ravel(), heatmap.ravel())


def get_q_iou(mask, heatmap, q=0.025):
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


def get_q_iou_stats(attrs, q=0.025):
    qiou_list = []
    for j, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, (_, mask) = sample
        qiou = get_q_iou(mask, attrs[j], q=q)
        qiou_list.append(qiou)
    return sum(qiou_list) / len(qiou_list)


if __name__ == "__main__":

    baseline_model_path = "/hdd0/projects/regex/models/unet_encoder_baseline_score=0.9717.pt"
    dual_model_path = ""

    calc_attrs = True
    calc_stats = True
    levels = [3, 4]
    resize_filter = Resize((256, 256), interpolation=InterpolationMode.NEAREST)

    # Base model and dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset, val_loader = get_dataset()
    base_model = get_base_model(baseline_model_path)

    print("*** Baseline methods ***")
    for level in levels:
        print(f"*** Level {level} ***")
        if calc_attrs:
            if not os.path.exists(f'attrs/gc_attrs_level={level}.pkl'):
                print(f"Calculating attributions: GracCAM level={level}")
                gc_attrs = get_gradcam_attr(base_model, level=level)
                with open(f'attrs/gc_attrs_level={level}.pkl', 'wb') as f:
                    pickle.dump(gc_attrs, f)
            if not os.path.exists(f'attrs/inxgrad_attrs_level={level}.pkl'):
                print(f"Calculating attributions: Gradient*Input level={level}")
                inxgrad_attrs = get_inxgrad_attr(base_model, level=level)
                with open(f'attrs/inxgrad_attrs_level={level}.pkl', 'wb') as f:
                    pickle.dump(inxgrad_attrs, f)
            if not os.path.exists(f'attrs/dl_attrs_level={level}.pkl'):
                print(f"Calculating attributions: DeepLIFT level={level}")
                dl_attrs = get_deeplift_attr(base_model, level=level)
                with open(f'attrs/dl_attrs_level={level}.pkl', 'wb') as f:
                    pickle.dump(dl_attrs, f)
            if not os.path.exists(f'attrs/lrp_attrs_level={level}.pkl'):
                print(f"Calculating attributions: LRP level={level}")
                lrp_attrs = get_lrp_attr(base_model, level=level)
                with open(f'attrs/lrp_attrs_level={level}.pkl', 'wb') as f:
                    pickle.dump(lrp_attrs, f)

        if calc_stats:
            with open(f'attrs/gc_attrs_level={level}.pkl', 'rb') as f:
                gc_attrs = pickle.load(f)
            with open(f'attrs/inxgrad_attrs_level={level}.pkl', 'rb') as f:
                inxgrad_attrs = pickle.load(f)
            with open(f'attrs/dl_attrs_level={level}.pkl', 'rb') as f:
                dl_attrs = pickle.load(f)
            with open(f'attrs/lrp_attrs_level={level}.pkl', 'rb') as f:
                lrp_attrs = pickle.load(f)

            base_methods = {
                "GradCAM": gc_attrs,
                "InputXGradients": inxgrad_attrs,
                "DeepLIFT": dl_attrs,
                "LRP": lrp_attrs,
            }

            for name, attrs in base_methods.items():
                print(f"Processing results: , method={name}, level={level}")
                if not os.path.exists(f'results/method={name}_level={level}__qiou_q=0.1.pkl'):
                    stats = get_q_iou_stats(attrs, q=0.1)
                    with open(f'results/method={name}_level={level}__qiou_q=0.1.pkl', 'wb') as f:
                        pickle.dump(stats, f)
                if not os.path.exists(f'results/method={name}_level={level}__qiou_q=0.05.pkl'):
                    stats = get_q_iou_stats(attrs, q=0.05)
                    with open(f'results/method={name}_level={level}__qiou_q=0.05.pkl', 'wb') as f:
                        pickle.dump(stats, f)
                if not os.path.exists(f'results/method={name}_level={level}__qiou_q=0.025.pkl'):
                    stats = get_q_iou_stats(attrs, q=0.025)
                    with open(f'results/method={name}_level={level}__qiou_q=0.025.pkl', 'wb') as f:
                        pickle.dump(stats, f)
                if not os.path.exists(f'results/method={name}_level={level}__map.pkl'):
                    stats = get_ap_stats(attrs)
                    with open(f'results/method={name}_level={level}__map.pkl', 'wb') as f:
                        pickle.dump(stats, f)

        if calc_attrs:
            if not os.path.exists(f'attrs/baseline_attrs_level={level}.pkl'):
                print(f"Calculating attributions: Baseline level={level}")
                baseline_attrs = get_baseline_attr(baseline_model_path, level=level)
                with open(f'attrs/baseline_attrs_level={level}.pkl', 'wb') as f:
                    pickle.dump(baseline_attrs, f)
            if not os.path.exists(f'attrs/dual_attrs_level={level}.pkl'):
                print(f"Calculating attributions: Dual level={level}")
                dual_attrs = get_dual_attr(dual_model_path, level=level)
                with open(f'attrs/dual_attrs_level={level}.pkl', 'wb') as f:
                    pickle.dump(dual_attrs, f)

        if calc_stats:
            with open(f'attrs/dual_attrs_level={level}.pkl', 'rb') as f:
                dual_attrs = pickle.load(f)

        ens_methods = {
            "Dual": dual_attrs
        }

        for name, attrs in ens_methods.items():
            print(f"Processing results: method={name}, level={level}")
            if not os.path.exists(f'results/method={name}_level={level}__qiou_q=0.1.pkl'):
                stats = get_q_iou_stats(attrs, q=0.1)
                with open(f'results/method={name}_level={level}__qiou_q=0.1.pkl', 'wb') as f:
                    pickle.dump(stats, f)
            if not os.path.exists(f'results/method={name}_level={level}__qiou_q=0.05.pkl'):
                stats = get_q_iou_stats(attrs, q=0.05)
                with open(f'results/method={name}_level={level}__qiou_q=0.05.pkl', 'wb') as f:
                    pickle.dump(stats, f)
            if not os.path.exists(f'results/method={name}_level={level}__qiou_q=0.025.pkl'):
                stats = get_q_iou_stats(attrs, q=0.025)
                with open(f'results/method={name}_level={level}__qiou_q=0.025.pkl', 'wb') as f:
                    pickle.dump(stats, f)
            if not os.path.exists(f'results/method={name}_level={level}__map.pkl'):
                stats = get_ap_stats(attrs)
                with open(f'results/method={name}_level={level}__map.pkl', 'wb') as f:
                    pickle.dump(stats, f)

