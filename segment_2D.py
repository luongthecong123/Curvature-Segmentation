import cv2
import matplotlib.pyplot as plt
import numpy as np

def test_one_image(path):
    img = cv2.imread(path)

    # preprocess image, remove border
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_mask = np.where((img_gray < 10) | (img_gray > 200), 0, 1)
    plt.imshow(img_mask, cmap='gray'); plt.title("Proprocess mask")
    # plt.savefig(r"D:\SIVOSSE\Cours\Project_SI\FINAL_REPORT\img\Mask.png", dpi=300)
    plt.show()
    
    # apply mask on the hue channel
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    img_masked = h*img_mask

    # hue channel thresholding
    h_new = np.where((img_masked>30) & (img_masked < 110), img_masked, 0)

    # plot results
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(10)
    ax1.imshow(img_rgb) # original rgb image
    ax1.set_title("The original image")
    ax2.imshow(h_new, cmap = 'hsv', vmin=0, vmax=179) # classified image
    ax2.set_title("Segmented ring")
    # plt.savefig(r"D:\SIVOSSE\Cours\Project_SI\FINAL_REPORT\img\ring.png", dpi=300)
    plt.show()

def trim(vid_path, start, stop):
    '''
    start, stop = "00:00:03", "00:00:10"
    return output_path
    '''
    import subprocess as sp
    import io
    import os

    ffmpegPath = r"D:\Programs\ffmpeg-full\bin\ffmpeg.exe"
    ffprobePath = r"D:\Programs\ffmpeg-full\bin\ffprobe.exe"
    check_dim = "-v error -select_streams v:0 -show_entries stream=width,height -of compact"
    stream = "v"
    
    path = os.path.dirname(vid_path)
    file_name = "trimmed_" + os.path.basename(vid_path)

    trim_path = os.path.join(path, file_name)

    # trimming
    trim_command = f"echo y| {ffmpegPath} -i {vid_path} -ss {start} -to {stop} -c copy {trim_path} "
    print(trim_command)
    # p = sp.Popen([trim_command, " -y"], shell=True, stdout=sp.PIPE) # reply yes to overwrite
    p = sp.Popen(trim_command, shell=True, stdout=sp.PIPE)

    return trim_path

def resize_HD(vid_path):
    import subprocess as sp
    import io
    import os

    ffmpegPath = r"D:\Programs\ffmpeg-full\bin\ffmpeg.exe"
    path = os.path.dirname(vid_path)
    file_name = "resized_" + os.path.basename(vid_path)
    resize_path = os.path.join(path, file_name)
    resize_command = f"echo y| {ffmpegPath} -i {vid_path} -filter:v scale=1280x720 {resize_path}"
    print(resize_command)
    p = sp.Popen(resize_command, shell=True, stdout=sp.PIPE)

    return resize_path

def get_hue(rgb):
    import torch

    rgb = (rgb/255.0).to(dtype=torch.float16)
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3  # if max - min == 0 => (0.4, 0.4, 0.4) (gray) => add another index = 3 to this position
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) *60)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta)*60 + 120)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta)*60 + 240)[cmax_idx == 2]
    hsl_h[cmax_idx == 3] = 0.
    # convert to range 0 -> 255 uint8, 360/2 = 180 -> new range: 0 -> 180 (openCV)
    hsl_h = (torch.where(hsl_h < 0, hsl_h + 360, hsl_h)/2.).to(dtype=torch.uint8)

    return hsl_h


def hue_trim(input_path, output_path="", start=0, end=100):
    '''
    Require ffmpeg
    start, end: percentage of desired portion of video to process
    '''
    import torch
    import torchvision
    import skvideo.io 

    
    videodata = skvideo.io.vread(input_path)  
    writer = skvideo.io.FFmpegWriter(output_path) # require numpy 1.19

    data_torch = torch.from_numpy(videodata)
    data_tvision = data_torch.permute(0, 3, 1, 2)
    print(data_tvision.shape)
    to_gray = torchvision.transforms.Grayscale(num_output_channels=1) #torchvision object
    data_tvision = list(torch.split(data_tvision, 32))
    print("type(data_tvision): ", type(data_tvision))


    start = int(start/100)*len(videodata)
    end = int(100/100)*len(videodata)
    for i, batch in enumerate(data_tvision[start:end]):
        '''
        batch of shape: 32,3,720,1280
        hue of shape: 32,1,720,1280
        '''
        batch = batch.cuda()
        
        data_gray = to_gray(batch)
        data_mask = torch.where((data_gray < 5) | (data_gray > 200), 0, 1).cuda()
        data_mask = batch*data_mask
        hue = get_hue(data_mask)

        # clipping the histogram
        hue = torch.where((hue>30) & (hue< 110), 255, 0).to(dtype=torch.uint8)
        hue = hue.tile((1,3,1,1))
        hue = torch.concat((batch.permute((0, 2, 3, 1)), hue.permute((0, 2, 3, 1))), dim=2)
        print(f"Batch {i+1} has shape {hue.shape}, dtype: {hue.dtype} memory: {torch.cuda.memory_allocated()/1024**3}")

        writer.writeFrame(hue.cpu().numpy())

    writer.close()

def write_video_demo():

    input_path = r"D:\SIVOSSE\Cours\Project_SI\Videos_21\2022-11-16_170457_VID004.mp4"
    input_path = r"D:\SIVOSSE\Cours\Project_SI\video_19\2022-02-09_143900_VID002.mp4"
    output_path = r"D:\SIVOSSE\Cours\Project_SI\results\total_full.mp4"


    # video trimming using ffmpeg, require ffmpeg installed on computer

    # trim_path = trim(input_path, start="00:00:21", stop="00:00:30")
    # print("-------------------------------------------------")
    # print(trim_path)
    # print("-------------------------------------------------")
    # resize_path = resize_HD(r"D:\SIVOSSE\Cours\Project_SI\video_19\trimmed_2022-02-09_143900_VID002.mp4")
    
    

    input_path = r"D:\SIVOSSE\Cours\Project_SI\video_19\resized_trimmed_2022-02-09_143900_VID002.mp4"
    input_path = r"D:\SIVOSSE\Cours\Project_SI\video_19\resized_2022-02-09_143519_VID001.mp4"
    output_path = r"D:\SIVOSSE\Cours\Project_SI\results\result_2022-02-09_143519_VID001_new2.mp4"
    hue_trim(input_path, output_path)




if __name__ == "__main__":

    # demo on one image
    path = "img/ring_2.jpg"
    test_one_image(path)

    # Demo on video, require ffmpeg installed. If error, install numpy = 1.19
    # write_video_demo()
