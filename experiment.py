import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path


def set_up_experiment(args, experiment, suffix=""):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    prefix_exp = get_prefix_exp(experiment=experiment, args=args)
    Path(prefix_exp).mkdir(parents=True, exist_ok=True)

    """ SAVE PARAMETERS """
    with open(f"{prefix_exp}/args.json", 'w') as file:
        json.dump(vars(args), file)

    return prefix_exp


def set_up_experiment_manual_prefix(args, prefix_exp):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    
    Path(prefix_exp).mkdir(parents=True, exist_ok=True)

    """ SAVE PARAMETERS """
    with open(f"{prefix_exp}/args.json", 'w') as file:
        json.dump(vars(args), file)

    return prefix_exp


def get_prefix_exp(experiment, args):
    if experiment == "gw_vae":
        return gw_vae_prefix_exp(args)
    elif experiment == "gw_vae_fixed_decoder":
        return gw_vae_fixed_decoder_prefix_exp(args)
    elif experiment == "gw_ae_fixed_decoder":
        return gw_ae_fixed_decoder_prefix_exp(args)
    elif experiment == "vae":
        return vae_prefix_exp(args)
    elif experiment == "wgan":
        return wgan_prefix_exp(args)
    elif experiment == "gw_ae":
        return gw_ae_prefix_exp(args)
    elif experiment == "gw_vae_gan":
        return gw_vae_gan_prefix_exp(args)
    elif experiment == "gw_vae_simple":
        return gw_vae_simple_prefix_exp(args)
    elif experiment == "gw_vae_gan_simple":
        return gw_vae_gan_simple_prefix_exp(args)
    elif experiment == "gwae_toy":
        return gwae_toy_prefix_exp(args)
    elif experiment == "gw_ae_gan_simple":
        return gw_ae_gan_simple_prefix_exp(args)
    else:
        raise NotImplementedError


def gw_vae_simple_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[gw_vae_simple]_" \
           f"[data={args.data}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"[dim_x={args.dim_x}]_" \
           f"[dim_y={args.dim_y}]_" \
           f"[dim_z={args.dim_z}]_" \
           f"[c_ortho={args.c_ortho}]_" \
           f"[c_prior={args.c_prior}]_" \
           f"[divergence={args.divergence}]_" \
           f"[adversary_type={args.adversary_type}]"


def gwae_toy_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[gwae_toy]_" \
           f"[data={args.data_x}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"[dim_x={args.dim_x}]_" \
           f"[dim_y={args.dim_y}]_" \
           f"[dim_z={args.dim_z}]_" \
           f"[c_prior={args.c_prior}]_" \
           f"[prior={args.prior}]_" \
           f"[divergence={args.divergence}]"


def gw_vae_gan_simple_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[gw_vae_gan_simple]_" \
           f"[{args.data_x}_to_{args.data_y}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"[dim_x={args.dim_x}]_" \
           f"[dim_y={args.dim_y}]_" \
           f"[dim_z={args.dim_z}]_" \
           f"[c_ortho={args.c_ortho}]_" \
           f"[c_prior={args.c_prior}]_" \
           f"[divergence={args.divergence}]_" \
           f"[adversary_type={args.adversary_type}]"


def gw_ae_gan_simple_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[gw_vae_gan_simple]_" \
           f"[{args.data_x}_to_{args.data_y}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"[dim_x={args.dim_x}]_" \
           f"[dim_y={args.dim_y}]_" \
           f"[dim_z={args.dim_z}]_" \
           f"[c_ortho={args.c_ortho}]_" \
           f"[c_prior={args.c_prior}]_" \
           f"[divergence={args.divergence}]_" \
           f"[adversary_type={args.adversary_type}]"


def wgan_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[wgan]_" \
           f"[data={args.data_y}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"[dim_z={args.dim_z}]_" \
           f"[dis_iter={args.dis_iter}]_" \
           f"[c_gp={args.c_gp}]"


def gw_vae_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[gw_vae]_" \
           f"[data={args.data_x}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"[dim_z={args.dim_z}]_" \
           f"[c_ortho={args.c_ortho}]_" \
           f"[c_prior={args.c_prior}]_" \
           f"[divergence={args.divergence}]_" \
           f"[adversary={args.adversary_type}]"


def gw_ae_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[gw_ae]_" \
           f"[data={args.data_x}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"[dim_z={args.dim_z}]_" \
           f"[c_ortho={args.c_ortho}]_" \
           f"[c_prior={args.c_prior}]_" \
           f"[divergence={args.divergence}]_" \
           f"[adversary={args.adversary_type}]"


def gw_vae_fixed_decoder_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[gw_vae_fixed_decoder]_" \
           f"[{args.data_x}_to_{args.data_y}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"{gw_vae_part(args, end=False)}"


def gw_vae_part(args, end=False):
    output = f"[dim_z={args.dim_z}]_" \
             f"[c_ortho={args.c_ortho}]_" \
             f"[c_nuclear={args.c_nuclear}]_" \
             f"[c_prior={args.c_prior}]_" \
             f"[c_log_var={args.c_log_var}]_" \
             f"[prior={args.prior}]_" \
             f"[divergence={args.divergence}]_" \
             f"[adversary_type={args.adversary_type}]_"

    if not end:
        output += "_"

    return output


def gw_ae_fixed_decoder_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[gw_ae_fixed_decoder]_" \
           f"[{args.data_x}_to_{args.data_y}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"[dim_z={args.dim_z}]_" \
           f"[c_ortho={args.c_ortho}]_" \
           f"[c_prior={args.c_prior}]_" \
           f"[divergence={args.divergence}]"


def gw_vae_gan_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[gw_vae_wgan]_" \
           f"[{args.data_x}_to_{args.data_y}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"[dim_z={args.dim_z}]_" \
           f"[dis_iter={args.dis_iter}]_" \
           f"[c_ortho={args.c_ortho}]_" \
           f"[c_gp={args.c_gp}]_" \
           f"[c_prior={args.c_prior}]_" \
           f"[divergence={args.divergence}]"


def vae_prefix_exp(args):
    return f"./experiment/[{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}]_" \
           f"[vae]_" \
           f"[data={args.data_x}]_" \
           f"[batch_size={args.batch_size}]_" \
           f"[dim_z={args.dim_z}]_" \
           f"[c_prior={args.c_prior}]_" \
           f"[divergence={args.divergence}]"
