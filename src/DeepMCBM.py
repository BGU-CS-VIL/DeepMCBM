import os
import train_STN
import train_BMN
import predict_BMN
import args as ARGS
import neptune.new as neptune

def main(args,run=None):
    if args.train_STN:
        print("\n### TRAINING STN START ###\n")
        ckpt_name = train_STN.main(args,run=run)
        args.STN_ckpt = ckpt_name+"_STN_best.ckpt"
        print('### checkpoint saved:',os.path.join(args.STN_ckpt_dir,args.STN_ckpt))
        print("\n### TRAINING STN DONE ###")

    if args.train_BMN:
        print("\n### TRAINING AE START ###\n")
        ckpt_name = train_BMN.main(args,run=run)
        args.BMN_ckpt = ckpt_name+"_BMN_best.ckpt"
        print('### checkpoint saved:',os.path.join(args.BMN_ckpt_dir,args.BMN_ckpt))
        print("### TRAINING AE DONE ###")

    print("\n### PREDICT AND EVALUATE MODEL START ###")
    predict_BMN.main(args,run=run)
    print("\n### PREDICT AND EVALUATE MODEL END ###")

    if run:
        run.stop()
        
if __name__ == "__main__":
    parser = ARGS.get_argparser()
    args = parser.parse_args()
    run = neptune.init(project=args.neptune_project,
                        api_token=args.neptune_api_token,
                        source_files=['*.py'],
                        tags=args.tags)
    if args.DryRun:
        args.STN_total_epochs = 3
        args.AE_total_epochs = 3 

    main(args,run=run)