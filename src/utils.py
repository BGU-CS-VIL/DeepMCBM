import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pandas as pd
# import Moment
import torch.nn.functional as F
import numpy.ma as ma
from typing import List
from transforms import affine_warp, homography_warp
from PIL import Image 
from sklearn import metrics
from natsort import natsorted
import pathlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fetch_Neptune_run_id(run):
    """This is a wrapper function to let vs code know there is a return value from the fetch function. 
        Otherwise the code after the fetch function is markes as unreachablke
    Args:
        run (Neptune run): the run in interest
    Returns:
        string : The id of run 
    """ 
    return run['sys/id'].fetch()
    
def extractImages(pathIn, pathOut, rescale=1):
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            hight, width, _ = image.shape
            image = cv2.resize(image, (int(width/rescale), int(hight/rescale)))
            cv2.imwrite(pathOut + "/frame%05d.jpg" %
                        (count), image)     # save frame as JPEG file
            count += 1
    if count > 0:
        print('%d images where extracted successfully' % count)
    else:
        print('Images extraction failed.')
    return count


def create_dummy_init_csv(dir_path):
    df = pd.read_csv("/home/ergu/videos/tennis/init_t.csv")
    columns = df.columns
    names = os.listdir(os.path.join(dir_path, "frames"))
    l = len(names)
    new_df = pd.DataFrame({columns[0]: names,
                           columns[1]: np.ones(l),
                           columns[2]: np.zeros(l),
                           columns[3]: np.zeros(l),
                           columns[4]: np.zeros(l),
                           columns[5]: np.ones(l),
                           columns[6]: np.zeros(l),
                           columns[7]: np.zeros(l),
                           columns[8]: np.zeros(l),
                           columns[9]: np.ones(l)})
    print(new_df.head)
    new_df.to_csv(os.path.join(dir_path, "init_t.csv"), index=False)


def cnn_layer_output_size(input, kernel, padding=0, stride=1, dilation=1):
    def out_size_cnn_(input, kernel, padding=1, stride=1, dilation=1):
        return int((input+2*padding-dilation*(kernel-1)-1)/stride+1)
    try:
        s0 = out_size_cnn_(input[0], kernel, padding, stride, dilation)
        s1 = out_size_cnn_(input[1], kernel, padding, stride, dilation)
        return (s0, s1)
    except:
        return out_size_cnn_(input, kernel, padding, stride, dilation)


def cnn_output_size(input_shape, kernels, strides):
    output_shape = input_shape
    for k, s in zip(kernels, strides):
        output_shape = cnn_layer_output_size(output_shape, kernel=k, stride=s)
    return output_shape


def out_size_Tcnn(input, padding, out_padding, kernel, stride, dilation=1):
    return (input-1)*stride-2*padding+dilation*(kernel-1)+out_padding+1


def show_tensor_image(image):
    image = prepare_image_to_show(image)
    plt.imshow(image)
    plt.show(block=True)


def save_image(path, image):
    image = prepare_image_to_show(image)
    plt.imsave(path, image.numpy())


def prepare_image_to_show(image):
    with torch.no_grad():
        image = image.cpu()
        if len(image.shape) > 3:
            image = image[0]
        if image.shape[0] == 1 or len(image.shape) == 2:
            image = image.expand((3, image.shape[-2], image.shape[-1]))
        image = torch.moveaxis(image, 0, -1)
        if image.min() < 0 or image.max() > 1:
            image = scale01(image)
        return image


def scale01(image):
    image = image-image.min()
    if image.max() != 0:
        image = image/image.max()
    return image


def weighted_average(mu, image_acc_sum, mask_acc_sum, zero_sensitivity=1e-5):
    # mu is the target tensor to hold the average
    mu.fill_(0.0)
    valid_index = mask_acc_sum >= zero_sensitivity
    mu[valid_index] = image_acc_sum[valid_index]/mask_acc_sum[valid_index]
    return mu


def log_image(image, title, run, neptun_dir):
    image = prepare_image_to_show(image)
    fig = plt.figure()
    plt.title(title)
    plt.imshow(image)
    run[neptun_dir].log(fig)
    # path = str.split(neptun_dir,"/")
    # dir = os.path.join(*path[0:-1])
    # pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
    # plt.savefig(os.path.join(*path)+".png")
    # plt.close()

def warp_img(img, theta_dict, shape):

    if "affine" in theta_dict.keys():
        grid = F.affine_grid(theta_dict["affine"], shape)
        img = F.grid_sample(img, grid)
    return img


def warp_inv(img, theta_dict, shape,device,global_transform=None):
    """ Warps image by inverse transformation theta_dict
    Args:
        img: Image to warp
        theta_dict: Dictionary of theta values
        shape: Shape of the image
    Returns:
        Warped image
    """
    N, C , H, W = shape
    grid = None
    if "homography" in theta_dict.keys():
        _, grid = homography_warp(theta_dict["homography"], shape, exp=False, grid=grid, inverse=True,device=device)
    if "affine" in theta_dict.keys():
        _, grid = affine_warp(theta_dict["affine"], shape, exp=True, grid=grid,global_transform=global_transform, inverse=True,device=device)
     
    # Intrepolate
    grid = grid.permute(0, 2, 1).reshape(-1, H, W, 2)  # for F.grid_sample
    img = F.grid_sample(img, grid)
    return img



def create_pixel_stack(data_loader, stn):
    image_stack = []
    mask_stack = []
    for i,image in enumerate(data_loader):
        image_out, mask_out, transform = stn(image.cuda())
        image_stack.append(image_out)
        mask_stack.append(mask_out)
        # # for debug
        # path = "./images"
        # img = prepare_image_to_show(image_out)
        # mask = prepare_image_to_show(mask_out)
        # plt.imsave(f"{path}/frames/image_in_{i}.png", img.numpy())
        # plt.imsave(f"{path}/masks/mask_in_{i}.png", mask.numpy())
        # # plt.imshow(img)
        # # plt.show()
        # print(transform)
        # # end debug

    image_stack = torch.cat(image_stack)
    mask_stack = torch.cat(mask_stack)
    return image_stack, mask_stack


def trim_pixel_stack_(image_stack, mask_stack, trim_percentage, zero_sensitivity=1e-5):
    low = trim_percentage
    high = 1 - trim_percentage
    invalid_mask = mask_stack <= zero_sensitivity
    image_stack = ma.masked_array(image_stack, mask=invalid_mask)
    sort_idx = image_stack.argsort(axis=0)
    image_stack[:] = np.take_along_axis(image_stack, sort_idx, axis=0)
    mask_stack[:] = np.take_along_axis(mask_stack, sort_idx, axis=0)

    count = mask_stack.sum(axis=0)
    low_bound = (count*low).astype(int)
    high_boung = (count*high).astype(int)

    for i in range(image_stack.shape[2]):
        for j in range(image_stack.shape[3]):
            # image
            image_stack[:low_bound[0, i, j], :, i, j] = 0
            image_stack[high_boung[0, i, j]:, :, i, j] = 0
            # mask
            mask_stack[:low_bound[0, i, j], :, i, j] = 0
            mask_stack[high_boung[0, i, j]:, :, i, j] = 0


def safe_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print(f"create dir : {dir_path}")
        return True
    else:
        print(f"dir already exist : {dir_path}")
        return False


def trim_pixel_stack(image_stack, mask_stack, trim_percentage):
    # move to numpy to use masked sorting
    image_stack = image_stack.cpu().numpy()
    mask_stack = mask_stack.cpu().numpy()
    # sort and zero firts and last trim_percentage % (in-place), cahnge mask correspondingly
    trim_pixel_stack_(image_stack,
                      mask_stack,
                      trim_percentage,
                      zero_sensitivity=0.1)
    # move back to tensors
    image_stack = torch.tensor(image_stack, device=device)
    mask_stack = torch.tensor(mask_stack, device=device)
    return image_stack, mask_stack

def trim_pixel_stack_(image_stack, mask_stack, trim_percentage, zero_sensitivity=1e-5):
    low = trim_percentage
    high = 1 - trim_percentage
    invalid_mask = mask_stack <= zero_sensitivity
    image_stack = ma.masked_array(image_stack, mask=invalid_mask)
    sort_idx = image_stack.argsort(axis=0)
    image_stack[:] = np.take_along_axis(image_stack, sort_idx, axis=0)
    mask_stack[:] = np.take_along_axis(mask_stack, sort_idx, axis=0)

    count = mask_stack.sum(axis=0)
    low_bound = (count*low).astype(int)[0]
    high_bound = (count*high).astype(int)[0]

    for i in range(image_stack.shape[2]):
        for j in range(image_stack.shape[3]):
            if high_bound[i,j] > low_bound[i,j]: 
                # image
                image_stack[:low_bound[i, j], :, i, j] = 0
                image_stack[high_bound[i, j]:, :, i, j] = 0
                # mask
                mask_stack[:low_bound[i, j], :, i, j] = 0
                mask_stack[high_bound[i, j]:, :, i, j] = 0



def frames_to_video(image_folder, video_name, speed):
    images = [name for name in os.listdir(image_folder)]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, speed, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()


def video_to_frames(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            hight, width, _ = image.shape
            cv2.imwrite(os.path.join(pathOut, f"frame{count:06}.png"), image)
            count += 1
    if count > 0:
        print('%d images where extracted successfully' % count)
    else:
        print('Images extraction failed.')
    return count


def panorama_fg(fg, pad, transform):
    fg = F.pad(fg,
               (0,           # left
                pad[1],      # right
                0,           # top
                pad[0]       # bottom
                ))
    fg = warp_img(fg, transform, fg.shape)
    return fg


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")


def calc_F_measure(fg_est,fg_gt):
    fg_est = fg_est
    fg_gt = fg_gt
    fg_est = np.where(fg_est > 0.5, 1, 0)
    fg_gt = np.where(fg_gt > 0.5, 1, 0)
    TP = np.sum(fg_est * fg_gt)
    FP = np.sum(fg_est) - TP
    FN = np.sum(fg_gt) - TP
    TN = np.sum(fg_est == 0) - FN
    
    # if TP == 0:
    #     return 0, 0, 0, 0, 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    F_measure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return F_measure, precision, recall, TPR, FPR
    

def calc_TPR_FPR(fg_est,fg_gt):
    """_summary_
    
    Parameters:
    fg_est: estimated foreground
    fg_gt: ground truth foreground

    Returns:
    TPR: True Positive Rate
    FPR: False Positive Rate
    """
    fg_est = np.where(fg_est > 0.5, 1, 0)
    fg_gt = np.where(fg_gt > 0.5, 1, 0)
    TP = np.sum(fg_est * fg_gt)
    FP = np.sum(fg_est) - TP
    FN = np.sum(fg_gt) - TP
    if TP == 0:
        return 0, 0
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TP)
    return TPR, FPR




def freeze_layers(model, layer_names: List[str]):
    """ Freeze layers in a model according to layer names.
        Freeze = requires_grad = False 
    """
    for layer_name in layer_names:
        print("freez:",layer_name)
        for param in model.named_parameters():
            if layer_name in param[0]:
                param[1].requires_grad = False


def unfreeze_layers(model, layer_names: List[str]):
    """ Unfreeze layers in a model according to layer names.
        Unfreeze = requires_grad = True 
    """
    for layer_name in layer_names:
        for param in model.named_parameters():
            if layer_name in param[0]:
                param[1].requires_grad = True


def print_and_write_to_file(str, file):
    file.write(str+"\n")
    print(str)

def save_model(save_path,model,optimizer,scheduler):
    checkpoint = {'state_dict': model.state_dict(), 
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()}
    torch.save(checkpoint, save_path)
    print("model checkpoint saved at : ", save_path)

def vec_to_perpective_matrix(vec):
    # For homography matrix
    # vec rep of the perspective transform has 8 dof; so add 1 for the bottom right of the perspective matrix;
    # note network is initialized to transformer layer bias = [1, 0, 0, 0, 1, 0] so no need to add an identity matrix here
    out = torch.cat((vec, torch.ones(
        (vec.shape[0], 1), dtype=vec.dtype, device=vec.device)), dim=1).reshape(vec.shape[0], -1)
    return out.reshape(-1, 3, 3)


def tree_view_video(frames1_dir,frames2_dir,frames3_dir,output_path,speed = 30):
    frames1 = [name for name in os.listdir(frames1_dir)]
    frames2 = [name for name in os.listdir(frames2_dir)]
    frames3 = [name for name in os.listdir(frames3_dir)]
    frames1.sort()
    frames2.sort()
    frames3.sort()

    sample_image = cv2.imread(os.path.join(frames1_dir, frames1[0]))
    height, width, layers = sample_image.shape
    height, width, layers = height*3, width, layers
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_path, fourcc, speed, (width, height))
    for image1,image2,image3 in zip(frames1,frames2,frames3):
        new_frame = np.zeros((height, width, layers), dtype=np.uint8)
        new_frame[:height//3, :, :] = cv2.imread(os.path.join(frames1_dir, image1)) # left top
        new_frame[height//3:2*height//3, : , :] = cv2.imread(os.path.join(frames2_dir, image2)) # right top
        new_frame[2*height//3: , :, :] = cv2.imread(os.path.join(frames3_dir, image3)) # left bottom
        video.write(new_frame)
    cv2.destroyAllWindows()
    video.release()

    
def update_special_args(args, csv_path="special_args.csv"):
    csv_path = os.path.join(os.getcwd(), csv_path)
    df = pd.read_csv(csv_path)
    dataset = args.dir
    new_args = df[df.dataset==dataset]
    if new_args.empty:
        return args
    # assign any variable which is not pd.nan
    for col in new_args.columns:
        if pd.notnull(new_args[col]):    
            val = new_args[col].values[0]
        
            if pd.notnull(val):
                setattr(args, col, val)
    return args

def fill_moments_no_trim(moments,STN,data_loader):
    with torch.no_grad():
        image = next(iter(data_loader))
        sample_image, sample_mask, transform = STN(image.cuda())
        moment_sum = torch.zeros_like(sample_image[0])
        mask_sum = torch.zeros_like(sample_mask[0])
        for i, m in enumerate(moments):
            for image in data_loader:
                    image_out, mask_out, transform = STN(image.cuda())
                    moment = (image_out-moments[0]*mask_out)**(i+1)
                    moment_sum+=moment.sum(dim=0)
                    mask_sum+=mask_out.sum(dim=0)
            weighted_average(
                moments[i], moment_sum, mask_sum, zero_sensitivity=1e-5)

def resize_results(datasets,methods,path):
    for dataset in datasets:
        print(dataset)
        for method in methods: 
            print(method)
            for img in os.listdir(os.path.join(path, dataset, method)):
                if img.endswith(".jpg"):
                    print(img)
                    img_path = os.path.join(path, dataset, method, img)
                    img = Image.open(img_path)
                    img_resize = img.resize((512,256))

                    reszie_image_path = img.filename.replace(".jpg", "_resize.png")
                    img_resize.save(reszie_image_path,"PNG")
                    print("resize image saved to: ", reszie_image_path)
                elif img.endswith(".png"):
                    print(img)
                    img_path = os.path.join(path, dataset, method, img)
                    img = Image.open(img_path)
                    img_resize = img.resize((512,256))

                    reszie_image_path = img.filename.replace(".png", "_resize.png")
                    img_resize.save(reszie_image_path,"PNG")
                    print("resize image saved to: ", reszie_image_path)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def get_concat_v(im1, im2,img3):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height + img3.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    dst.paste(img3, (0, im1.height+im2.height))
    return dst

def input_bg_frames(input_dir,bg_dir,output_path):
    input_frames = [name for name in os.listdir(input_dir)]
    bg_frames = [name for name in os.listdir(bg_dir)]
    input_frames.sort()
    bg_frames.sort()

    assert len(input_frames) == len(bg_frames) , f"input({len(input_frames)}) and bg frames length are not equal"

    for i,(bg_frame,input_frame) in enumerate(zip(bg_frames,input_frames)):
        bg_image = Image.open(os.path.join(bg_dir, bg_frame)).convert('RGB')
        input_image = Image.open(os.path.join(input_dir,input_frame)).convert('RGB')
        input_image = input_image.resize(bg_image.size)
        get_concat_v(input_image, bg_image).save(os.path.join(output_path, f'frame{i:05}.png'), 'PNG')


def input_bg_fg_frames(input_dir,bg_dir,fg_dir,output_path):
    input_frames = [name for name in os.listdir(input_dir)]
    bg_frames = [name for name in os.listdir(bg_dir)]
    fg_frames = [name for name in os.listdir(fg_dir)]

    input_frames.sort()
    bg_frames.sort()
    fg_frames.sort()

    assert len(input_frames) == len(bg_frames) , f"input({len(input_frames)}) and bg frames length are not equal"

    for i,(bg_frame,fg_frame,input_frame) in enumerate(zip(bg_frames,fg_frames,input_frames)):
        bg_image = Image.open(os.path.join(bg_dir, bg_frame)).convert('RGB')
        fg_image = Image.open(os.path.join(fg_dir, fg_frame)).convert('RGB')
        fg_image = fg_image.resize(bg_image.size)
        input_image = Image.open(os.path.join(input_dir,input_frame)).convert('RGB')
        input_image = input_image.resize(bg_image.size)
        get_concat_v(input_image, bg_image,fg_image).save(os.path.join(output_path, f'frame{i:05}.png'), 'PNG')


def image_list(dir_name):
    image_list = os.listdir(dir_name)
    image_list_paths = []
    for image_name in image_list:
        if not image_name.endswith((".avi", ".mp4", ".txt")):
            image_path = os.path.join(dir_name, image_name)
            image_list_paths.append(image_path)
    image_list_paths=natsorted(image_list_paths)
    return image_list_paths