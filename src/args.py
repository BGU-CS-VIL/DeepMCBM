import argparse

DATA_DIR = "./input"
OUTPUT_DIR = "./output"
CKPT_DIR = "./checkpoints"
MCBM_CKPT = "tennis.ckpt"
SEQUENCE = "tennis"


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DryRun', dest='DryRun', action='store_true')
    parser.set_defaults(DryRun=False)
    # logging args
    parser.add_argument("--log_name", type = str ,default="my_run")
    parser.add_argument("--tags", nargs ='+' ,type=str, default=["debug"])
    
    # dataset params
    parser.add_argument("--parent_dir", type=str, default=DATA_DIR)
    parser.add_argument("--dir", type=str, default=SEQUENCE)
    parser.add_argument("--source_shape", nargs ='+' ,type=int, default=(480,854))
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument('--outputs_dir', type=str, default='/vildata/guy/competitors_Results/')
    parser.add_argument('--method', type=str, default='MCBM')
    
    parser.add_argument('--CDNet', dest='CDNet', action='store_true')
    parser.set_defaults(CDNet=False)

    
    # panorama params
    parser.add_argument("--pad", nargs ='+' ,type=int, default=(700,700))
    parser.add_argument("--t", nargs ='+' ,type=float, default=(-0.5,-0.5))
    parser.add_argument("--mask_shape", nargs ='+' ,type=int, default=(256,512)) # change name to something more descriptive
    
    # STN params
    parser.add_argument("--memory", type=float, default=0.9)
    parser.add_argument("--TG", type=str,help= "TG: transformation group to use for alignment, choose Affine or Homo" ,default='Affine')
    parser.add_argument("--load_Affine", dest="load_Affine", action="store_true")
    parser.set_defaults(load_Affine=False)

    # STN training params
    parser.add_argument("--STN_batch_size", type=int, default=16)
    parser.add_argument("--STN_total_epochs", type=int, default=7000)
    parser.add_argument("--STN_lr", type=float, default=0.005)
    parser.add_argument("--train_STN", dest="train_STN", action="store_true")
    parser.add_argument("--no_train_STN", dest="train_STN", action="store_false")
    parser.set_defaults(train_STN=True)

    ## STN scheduler args 
    parser.add_argument("--STN_schedul_type", type=str, default='step', choices=['step','Mstep'])
    parser.add_argument("--STN_schedul_milestones", nargs ='+' ,type=int, default=[500])
    parser.add_argument("--STN_schedul_step", type=int, default=1000)
    parser.add_argument("--STN_schedul_gamma", type=float, default=0.5)
    
    ## STN optimizer args
    parser.add_argument("--STN_optimizer", type=str, default='SGD', choices=['SGD','Adam'])
    parser.add_argument("--STN_weight_decay", type=float, default=1e-5)
    parser.add_argument("--STN_momentum", type=float, default=0.9)

    ## STN loss args
    parser.add_argument("--beta", type=float, default=0.35,
                        help = "SmoothL1Loss parameter, defines the regions for L1 and L2 loss,\n used in the STN alignment loss" )
    
    # STN homography args
    # for init
    parser.add_argument('--homography',dest='homography', action='store_true')
    parser.set_defaults(homography=False)
    # for forward pass
    parser.add_argument("--train_homography_epoch", type=int, default=3000)

    # STN ckpt args
    parser.add_argument("--STN_ckpt_dir", type=str, default=CKPT_DIR)
    parser.add_argument("--STN_ckpt", type=str, default="tennis_AL-1954_best.ckpt")

    # AE model params
    parser.add_argument("--zero_moments", dest="zero_moments", action="store_true")
    parser.add_argument("--moments_num", type=int, default=2)
    parser.add_argument("--C", type=int, default=2)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--code_size",type=int,default=4)
    parser.add_argument("--cond_decoder",dest='cond_decoder',action='store_true')
    parser.add_argument("--no_cond_decoder",dest='cond_decoder',action='store_false')
    parser.set_defaults(cond_decoder=True)
    parser.add_argument("--AE_type", type=str, default='GDN', choices=['GDN','simple_conv'])

    # AE training params
    parser.add_argument("--AE_batch_size", type=int, default=16)
    parser.add_argument("--AE_total_epochs", type=int, default=2000)
    parser.add_argument("--AE_lr", type=float, default=0.0001)
    
    ## AE scheduler args 
    parser.add_argument("--AE_schedul_type", type=str, default='step', choices=['step','Mstep'])
    parser.add_argument("--AE_schedul_milestones", nargs ='+' ,type=int, default=[500])
    parser.add_argument("--AE_schedul_step", type=int, default=500)
    parser.add_argument("--AE_schedul_gamma", type=float, default=0.8)
    
    ## AE optimizer args
    parser.add_argument("--AE_optimizer", type=str, default='Adam', choices=['SGD','Adam'])
    parser.add_argument("--AE_weight_decay", type=float, default=1e-5)
    parser.add_argument("--AE_momentum", type=float, default=0.9)

    # AE loss args
    parser.add_argument("--AE_loss_type", type=str, default='GM', choices=['SL1','MSE','GM'])
    parser.add_argument("--sigma", type=float, default=0.05,
                        help = "geman mcclure parameter, defines the regions for L2 and constant loss, used in the BMN reconstruction loss" )

    # BMN params 
    parser.add_argument("--BMN_ckpt_dir", type=str, default=CKPT_DIR)
    parser.add_argument("--BMN_ckpt", default=MCBM_CKPT, type=str)
    parser.add_argument('--train_BMN', dest='train_BMN', action='store_true')
    parser.add_argument('--no_train_BMN', dest='train_BMN', action='store_false')
    parser.set_defaults(train_BMN=True)
    parser.add_argument('--trim_moments', dest='trim_moments', action='store_true')
    parser.add_argument('--no_trim_moments', dest='theta_embedding', action='store_false')
    parser.set_defaults(trim_moments=True)
    parser.add_argument("--trim_percentage", type=float, default=0.30)
    parser.add_argument("--Results_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument('--theta_embedding', dest='theta_embedding', action='store_true')
    parser.add_argument('--no_theta_embedding', dest='theta_embedding', action='store_false')
    parser.set_defaults(theta_embedding=False)


    # metrics
    parser.add_argument("--fg_threshold_min", type=float, default=0.005)
    parser.add_argument("--fg_threshold_max", type=float, default=0.99)
    parser.add_argument("--fg_threshold_step", type=float, default=0.05)
    parser.add_argument('--no_metrics', dest='calc_metrics', action='store_false')
    parser.add_argument('--metrics', dest='calc_metrics', action='store_true')
    parser.set_defaults(calc_metrics=True)

    parser.add_argument('--pretrain_resnet', dest='pretrain_resnet', action='store_true')
    parser.set_defaults(pretrain_resnet=False)
    parser.add_argument('--tess', dest='tess', action='store_true') # ask Ron if this is needed

    return parser