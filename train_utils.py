import torch
from sklearn.metrics import f1_score, balanced_accuracy_score


def compute_y(stats):
    # Compute y_hat / y_pred
    y_true, y_hat = zip(*stats['pred'])
    y_true, y_hat = torch.tensor(y_true, dtype=torch.float32), torch.tensor(y_hat, dtype=torch.float32)
    if any(y_hat.sum(dim=1) != 1.0):
        y_hat = torch.softmax(y_hat, dim=1)
    y_hat_max, y_pred = torch.max(y_hat, dim=1)
    y_pred = y_pred.float()

    y_true = y_true.numpy()
    y_hat = y_hat.numpy()
    y_hat_max = y_hat_max.numpy()
    y_pred = y_pred.numpy()

    return y_true, y_hat, y_hat_max, y_pred


def compute_stats_classification(stats, ret_metric="balanced_accuracy_score"):
    avg_val_total_loss = sum(stats['total_loss']) / len(stats['total_loss'])

    val_y_true, val_y_hat, val_y_hat_max, val_y_pred = compute_y(stats)

    output = [avg_val_total_loss]
    if ret_metric.lower() == "balanced_accuracy_score":
        output.append(balanced_accuracy_score(val_y_true, val_y_pred))
    elif ret_metric.lower() == "macro_f1_score":
        output.append(f1_score(val_y_true, val_y_pred, average="macro", zero_division=0))
    else:
        raise NotImplementedError

    return tuple(output)


def log_stats_classification(stats, outputs, targets, losses, batch_size=None, lr=None):
    lamb, kl_losses = None, []
    if len(losses) == 1:
        ce_loss = total_loss = losses[0]
    elif len(losses) == 3:
        ce_loss, kl_losses, total_loss = losses
        kl_losses = [k.detach() for k in kl_losses]
    else:
        raise ValueError(f"Cannot unpack losses: {losses}")
    ce_loss = ce_loss.item() / batch_size
    if 'cross_entropy_loss' in stats:
        stats['cross_entropy_loss'].append(ce_loss)
    else:
        stats['cross_entropy_loss'] = [ce_loss]
    for i, kl_loss in enumerate(kl_losses):
        if kl_loss is not None:
            kl_loss = kl_loss.item() / batch_size
            if f'kl_loss{i+1}' in stats:
                stats[f'kl_loss{i+1}'].append(kl_loss)
            else:
                stats[f'kl_loss{i+1}'] = [kl_loss]
    if lamb is not None:
        stats['lamb'] = lamb
    else:
        total_loss = losses[0]

    total_loss = total_loss.item() / batch_size
    if 'total_loss' in stats:
        stats['total_loss'].append(total_loss)
    else:
        stats['total_loss'] = [total_loss]

    np_outputs = outputs.detach().cpu().numpy().tolist()
    np_targets = targets.detach().cpu().numpy().tolist()
    preds = [(y, p) for (y, p) in zip(np_targets, np_outputs)]
    if 'pred' in stats:
        stats['pred'] += preds
    else:
        stats['pred'] = preds
    if lr:
        stats['lr'] = lr