import sys

sys.path.append("../")
from dataset_utils.preprocessing import letterbox_image_padded

import cv2
import time
import os
from PIL import Image
import numpy as np

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("FPS: ", fps)
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

def frames_to_video(inputpath,outputpath,fps):
   image_array = []
   files = [f for f in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, f))]
   files.sort(key = lambda x: int(x[:-4]))
   for i in range(len(files)):
       img = cv2.imread(inputpath + files[i])
       size =  (img.shape[1],img.shape[0])
       img = cv2.resize(img,size)
       image_array.append(img)
   # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
   fourcc = cv2.VideoWriter_fourcc(*'XVID')
   out = cv2.VideoWriter(outputpath,fourcc, fps, size)
   for i in range(len(image_array)):
       out.write(image_array[i])
   out.release()

def compress_video(inputpath, outputpath, fps):
    files = [f for f in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, f))]
    files.sort(key=lambda x: int(x[:-4]))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputpath, fourcc, fps, (416, 416))

    for i in range(len(files)):
        input_img = Image.open(inputpath + files[i])
        x_query, x_meta = letterbox_image_padded(input_img, size=(416, 416))

        x = (x_query[0] * 255.).astype(np.uint8)

        out.write(x[...,::-1])
    out.release()

if __name__=="__main__":
    # input_loc = 'frame2video/test_DFI_W.MP4'
    # output_loc = 'test_frame_W/'
    # video_to_frames(input_loc, output_loc)

    inputpath = 'haze_frame_Z/'
    outpath = 'frame2video/haze_DFI_Z.MP4'
    fps = 29 # MOT:25
    frames_to_video(inputpath, outpath, fps)

    # inputpath = 'video2frame_Z/'
    # # outpath = 'frame2video/test_DFI_W.MP4'
    # outpath = 'frame2video/test_DFI_Z.MP4'
    # fps = 29 # MOT:25
    # compress_video(inputpath, outpath, fps)