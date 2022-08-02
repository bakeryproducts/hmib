import torch


def default_zero_tensor(): return lambda: torch.tensor([0.])


@torch.no_grad()
def weight_watcher(cb):
    if cb.cfg.PARALLEL.IS_MASTER:
        tot = 0
        enc = 0
        dec = 0
        for k,v in cb.L.model.named_parameters():
            if 'bn' not in k:
                w = (v.detach()**2).sum()
                tot = tot + w
                if 'encoder' in k: enc += w
                if 'decoder' in k: dec += w
                # cb.L.writer.add_scalar(f'weights/{k}', w, cb.L.n_epoch)

        cb.L.writer.add_scalar('weights_total/sum', tot, cb.L.n_epoch)
        cb.L.writer.add_scalar('weights_total/enc', enc, cb.L.n_epoch)
        cb.L.writer.add_scalar('weights_total/dec', dec, cb.L.n_epoch)


def set_dropout(model, drop_rate):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def freeze_stages(model, stages, freeze=False):
    parts = []
    mm = model.model
    for k,v in mm.named_parameters():
        i = any([k.startswith(s) for s in stages])
        if i:
            v.requires_grad_(not freeze)
            parts.append(k)
    return parts
