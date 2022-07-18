import os
import train_STN
import train_BMN
import predict_BMN
import args as ARGS

# Ver_0
def main(args):
    if args.train_STN:
        print("\n### TRAINING STN START ###\n")
        ckpt_name = train_STN.main(args)
        args.STN_ckpt = ckpt_name+"_best.ckpt"
        print("\n### TRAINING STN DONE ###")
        print('### checkpoint saved:',os.path.join(args.STN_ckpt_dir,args.STN_ckpt))


    if args.train_BMN:
        print("\n### TRAINING AE START ###\n")
        ckpt_name = train_BMN.main(args)
        args.BMN_ckpt = ckpt_name+"_best.ckpt"
        print("\n### TRAINING AE DONE ###")
        print('### checkpoint saved:',os.path.join(args.BMN_ckpt_dir,args.BMN_ckpt))

    print("### PREDICT AND EVALUATE MODEL ###")
    predict_BMN.main(args)


if __name__ == "__main__":
    parser = ARGS.get_argparser()
    args = parser.parse_args()
    # args.tags = ["dry run","larger Homo reg"]
    # args.STN_total_epochs = 10
    # args.AE_total_epochs = 10
    # args.TG = "Homo"
    # args.load_Affine = True
    # args.BMN_ckpt ="horsejump-high_BAC-1048_last.ckpt"
    # args.train_STN = False
    # args.train_BMN = False
    # args.pad = [1500,1500]
    main(args)