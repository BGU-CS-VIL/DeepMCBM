import neptune.new as neptune
from utils import update_special_args
from metrics import calc_metric_and_MSE
import args as ARGS
import os

BW_METHODS = ["PRAC", "DECOLOR"]
METHODS = ["PRAC", "DECOLOR", "PCP_PTI", "PRPCA"] #JA-POLS
CDNET_DATA = ["zoomInZoomOut", "continuousPan", "sidewalk"]

def main(args):
    """
    Main function
    # outputs_dir: vildata/guy/competitors_Results/*args.dir*/*args.method*
    for instance: /vildata/guy/competitors_Results/tennis/JA-POLS
    """

    gt_path = os.path.join(args.parent_dir, args.dir, "GT")
    video_path = os.path.join(args.parent_dir, args.dir, "frames")
    bg_path = os.path.join(args.outputs_dir, args.dir, args.method, "bg")
    mse_path = os.path.join(args.outputs_dir, args.dir, args.method, "MSE")

    calc_metric_and_MSE(video_path=video_path, bg_path=bg_path,
                        gt_path=gt_path, mse_path=mse_path, args=args, method=args.method)
    


if __name__ == "__main__":
    parser = ARGS.get_argparser()
    args = parser.parse_args()
    datasets_list = os.listdir(args.outputs_dir)
    datasets_list = ["flamingo_test","flamingo_train"]
    for dataset in datasets_list:
    #for dataset in CDNET_DATA:
    #for dataset in CDNET_DATA:
        methods_list = os.listdir(os.path.join(args.outputs_dir, dataset))
        for method in methods_list:
            args.method = method

            #if dataset == "sidewalk":
            if dataset in CDNET_DATA:
                args.CDNet = True
            else:
                args.CDNet = False

            args.dir = dataset
            try:
                main(args)
            except Exception as e:
                print("Failed on dataset: {} and method: {}".format(dataset, method))
                continue
                
            print("Done!")
