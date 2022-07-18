import os
import train_STN
import train_BMN
import predict_BMN
import args as ARGS

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
    # for a dry run uncomment this
    args.STN_total_epochs = 10
    args.AE_total_epochs = 10 
    main(args)