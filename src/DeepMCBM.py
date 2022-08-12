import os
import train_STN
import train_BMN
import predict_BMN
import args as ARGS
import neptune.new as neptune

def main(args):
    if args.train_STN:
        print("\n### TRAINING STN ###\n")
        ckpt_name = train_STN.main(args)
        args.STN_ckpt = ckpt_name+"_STN_best.ckpt"
        print('### checkpoint saved:',os.path.join(args.STN_ckpt_dir,args.STN_ckpt))
        print("\n### TRAINING STN DONE ###")

    if args.train_BMN:
        print("\n### TRAINING CAE ###\n")
        ckpt_name = train_BMN.main(args)
        args.BMN_ckpt = ckpt_name+"_BMN_best.ckpt"
        print('### checkpoint saved:',os.path.join(args.BMN_ckpt_dir,args.BMN_ckpt))
        print("### TRAINING CAE DONE ###")

    print("\n### PREDICT and EVALUATE MODEL ###")
    predict_BMN.main(args)
    print("\n### PREDICT and EVALUATE MODEL END ###")
    
if __name__ == "__main__":
    parser = ARGS.get_argparser()
    args = parser.parse_args()
    if args.DryRun:
        args.STN_total_epochs = 3
        args.AE_total_epochs = 3 
    main(args)