"""Test pre-trained RGB model on a single video.

Date: 01/15/18
Authors: Bolei Zhou and Alex Andonian

This script accepts an mp4 video as the command line argument --video_file
and averages ResNet50 (trained on Moments) predictions on num_segment equally
spaced frames (extracted using ffmpeg).

Alternatively, one may instead provide the path to a directory containing
video frames saved as jpgs, which are sorted and forwarded through the model.

ResNet50 trained on Moments is used to predict the action for each frame,
and these class probabilities are average to produce a video-level predction.

Optionally, one can generate a new video --rendered_output from the frames
used to make the prediction with the predicted category in the top-left corner.

"""

import os
import re
import cv2
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
import moviepy.editor as mpy

import torch.optim
import torch.nn.parallel
from torch.nn import functional as F
from torch.autograd import Variable
import time
from test_model import load_model, load_categories, load_transform

import torch.nn as nn


def extract_frames(video_file, num_frames=1):
    """Return a list of PIL image frames uniformly sampled from an mp4 video."""
    # try:
    #     os.makedirs(os.path.join(os.getcwd(), 'frames'))
    # except OSError:
    #     pass

    # output = subprocess.Popen(['ffmpeg', '-i', video_file],
    #                           stderr=subprocess.PIPE).communicate()

    # # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
    # re_duration = re.compile('Duration: (.*?)\.')
    # duration = re_duration.search(str(output[1])).groups()[0]

    # seconds = functools.reduce(lambda x, y: x * 60 + y,
    #                            map(int, duration.split(':')))
    # rate = num_frames / float(seconds)

    # output = subprocess.Popen(['ffmpeg', '-i', video_file,
    #                            '-vf', 'fps={}'.format(rate),
    #                            '-vframes', str(num_frames),
    #                            '-loglevel', 'panic',
    #                            'frames/%d.jpg']).communicate()
    # mask_dir = './frames'
    # if not os.path.exists(mask_dir):
    #    os.makedirs(mask_dir)
    base_image_path = "/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d/Moments_in_Time_Raw/"
    end_image_path = "/".join(video_file.split("/")[-3:])
    
    image_folder_path = base_image_path + end_image_path
    print(image_folder_path)
    

    frame_paths = []
    for frame in os.listdir(image_folder_path):
        if 'jpg' in frame:
            frame_paths.append(os.path.join(image_folder_path, frame))

 
    frame_paths = sorted(frame_paths)

    start_time = time.time()
    frames = load_frames(frame_paths, args.num_segments)
   
    
    # subprocess.call(['rm', '-rf', 'frames'])
    return frames


def load_frames(frame_paths, num_frames=15):
    """Load PIL images from a list of file paths."""
    frames = [Image.open(frame).convert('RGB') for frame in frame_paths]
    if len(frames) == num_frames:
        pass # all is well
    elif len(frames) > num_frames:
        print("AHHH TOO BIG!")
        num_frames_remove = len(frames) - 15
        frames = frames[num_frames_remove:] #remove from start
    else:   
        print("AHHHHHHH TOO SMALL")
        num_pad_frames = num_frames - len(frames)
        for i in range(num_pad_frames):
            new_frame = frames[-1]
            frames.append(new_frame)

    return frames

def render_frames(frames, prediction):
    """Write the predicted category in the top-left corner of each frame."""
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / args.num_segments)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames


# options
parser = argparse.ArgumentParser(description="test TRN on a single video")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--video_file', type=str, default=None)
group.add_argument('--frame_folder', type=str, default=None)
parser.add_argument('--rendered_output', type=str, default=None)
parser.add_argument('--num_segments', type=int, default=15)
args = parser.parse_args()

# args.video_file = "/media/moumita/fce9875a-a5c8-4c35-8f60-db60be29ea5d/Moments_in_Time_256x256_30fps/validation/"

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor,self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x


def main():

    start_time = time.time()
    # Get dataset categories
    categories = load_categories()

    # Load RGB model
    model_id = 1
    model = load_model(model_id, categories).cuda()

    # Load the video frame transform
    transform = load_transform()

    mode_folder = args.video_file.split("/")[-1]
    if not mode_folder:
        raise Exception("DONT GIMME THAT LAST SLASH IN THE VIDEO_FILE plz.")
    feature_dir = "/media/lili/fce9875a-a5c8-4c35-8f60-db60be29ea5d/extracted_features_moments_raw/%s" % mode_folder

    if not os.path.exists(feature_dir):
       os.makedirs(feature_dir)
    for subdir in sorted(os.listdir(args.video_file)):
        print("subdir :", subdir)
        video_dir = os.path.join(args.video_file, subdir)

        logits_list = []
        features_list = []
        video_name_list = []

        for video_name in sorted(os.listdir(video_dir)):
            

            single_video_path = os.path.join(video_dir, video_name)
           
            frames = extract_frames(single_video_path, args.num_segments)

            # Prepare input tensor
            data = torch.stack([transform(frame) for frame in frames])

            input_var = Variable(data.view(-1, 3, data.size(2), data.size(3)),
                                 volatile=True).cuda()

            # Extract features before the fully connected layers
            res50_before_fc = FeatureExtractor(model)

            # Make video prediction
            logits = model(input_var)
            features = res50_before_fc(input_var)

            logits_np= logits.data.cpu().numpy()
            logits_list.append(logits_np)

            # save features before the fully connected layer

            features_list.append(np.squeeze(features.data.cpu().numpy()))

            stored_video_name = os.path.join(subdir, video_name)
            video_name_list.append(stored_video_name)
     
        np.save(os.path.join(feature_dir,"{}_logits.npy".format(subdir)), np.asarray(logits_list))
        np.save(os.path.join(feature_dir,"{}_names.npy".format(subdir)), np.asarray(video_name_list))
        np.save(os.path.join(feature_dir,"{}_features.npy".format(subdir)), np.asarray(features_list))


if __name__== "__main__":
  main()

