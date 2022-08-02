#!/usr/bin/env python3
import subprocess

import hydra


def to_str(cfg):
    ss = ''
    for k,v in cfg.items():
        if k.startswith('hydra'):
            continue
        if isinstance(v, int) or isinstance(v, float):
            ss+=f" ++{k}={v} "
            continue

        s = f"++{k}.{v}"
        for r in [" ", "{", "}", "'"]:
            s = s.replace(r, "")
        s = s.split(':')
        s, v = s[:-1], s[-1]
        s = '.'.join(s) + f'={v} '
        ss += s
    return ss


@hydra.main(config_path="", config_name="starter",version_base=None)
def main(cfg):
    cfg = dict(cfg)
    nproc = cfg.pop('nproc')
    args = to_str(cfg)
    # print(cfg, args)
    cmd = f'python3  -m torch.distributed.launch --use_env --nproc_per_node={nproc} src/main.py {args}'
    # cmd = f'ls'
    print(cmd)
    # cmd_single = f'python3 src/main.py --cfg {args.cfg}'
    # cmd = cmd_dist if cfg.PARALLEL.DDP else cmd_single
    p = subprocess.Popen(cmd, shell=True)
    p.communicate()


if __name__ == "__main__":
    main()
