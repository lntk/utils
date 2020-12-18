from datetime import datetime


def add_basic_arguments(parser):
    parser.add_argument('-name', default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    parser.add_argument('-seed', type=int, default=42)

    parser.add_argument('-data_dir', default="./data")

    parser.add_argument('-batch_size', type=int, default=1000)
    parser.add_argument('-test_batch_size', type=int, default=1000)
    parser.add_argument('-iterations', type=int, default=200000)
    parser.add_argument('-save_iter', type=int, default=1000)

    return parser


def add_toy_arguments(parser):
    parser.add_argument('-data_x', default="plane2d", help="[gaussian25|plane2d|scurve|swissroll]")
    parser.add_argument('-data_y', default="scurve", help="[gaussian25|plane2d|scurve|swissroll]")

    parser.add_argument('-dim_x', type=int, default=3)
    parser.add_argument('-dim_y', type=int, default=3)

    parser.add_argument('-noise', type=float, default=0.05)

    return parser


def add_vae_arguments(parser):
    parser.add_argument('-lr_vae', type=float, default=0.0002)
    parser.add_argument('-dim_z', type=int, default=2)

    parser.add_argument('-c_prior', type=float, default=1, help="coefficient for latent-distributional matching")

    parser.add_argument('-divergence', default="kl", help="[kl|mmd|wd|swd]")
    parser.add_argument('-prior', default="gaussian", help="[uniform|gaussian]")
    parser.add_argument('-recon_loss', default="mse", help="[mse|bce]")

    return parser


def add_image_arguments(parser):
    parser.add_argument('-data_x', default="mnist", help="[mnist(64)|fmnist(64)|svhn]")
    parser.add_argument('-data_y', default="svhn", help="[mnist(64)|fmnist(64)|svhn]")

    parser.add_argument('-size_x', type=int, default=32)
    parser.add_argument('-size_y', type=int, default=32)
    parser.add_argument('-channel_x', type=int, default=1)
    parser.add_argument('-channel_y', type=int, default=3)

    return parser


def add_gw_vae_arguments(parser):
    parser.add_argument('-lr_enc', type=float, default=0.0002)
    parser.add_argument('-lr_dec', type=float, default=0.0002)
    parser.add_argument('-lr_adv', type=float, default=0.0002)

    parser.add_argument('-dim_z', type=int, default=2)

    parser.add_argument('-n_enc', type=int, default=5)
    parser.add_argument('-n_dec', type=int, default=5)
    parser.add_argument('-n_adv', type=int, default=5)

    parser.add_argument('-dim_adv', type=int, default=3)
    parser.add_argument('-adv_iter', type=int, default=10)

    parser.add_argument('-c_ortho', type=float, default=32, help="coefficient for orthogonal Procrustes regularization to avoid metric maps collapsing")
    parser.add_argument('-c_prior', type=float, default=1, help="coefficient for latent-distributional matching")
    parser.add_argument('-c_log_var', type=float, default=0.1, help="coefficient for log-variance regularization in https://arxiv.org/abs/1802.03761")

    parser.add_argument('-c_nuclear', type=float, default=0, help="coefficient for nuclear norm of the encoder output to flatten the latent manifold")

    parser.add_argument('-prior', default="gaussian", help="[uniform|gaussian]")
    parser.add_argument('-divergence', default="mmd", help="[kl|mmd|wd|swd]")

    parser.add_argument('-adversary_type', default="fixed_x", help="[none|same|different|joint|fixed_x|fixed_y]")

    parser.add_argument('--normalized', default=False, action="store_true")
    parser.add_argument('--no_adversarial', default=False, action="store_true")

    return parser


def add_wgan_gp_arguments(parser):
    parser.add_argument('-lr_dis', type=float, default=0.0002)
    parser.add_argument('-dis_iter', type=int, default=5)
    parser.add_argument('-c_gp', type=float, default=1)

    return parser
