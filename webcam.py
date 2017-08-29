from __future__ import division, print_function

import argparse
import cv2
import numpy as np
from utils import preserve_colors
import tensorflow as tf
from imutils.video import FPS
from threading import Thread
import os
from utils import get_files, get_img, get_img_crop
from scipy.ndimage.filters import gaussian_filter
from coral import coral
from test import AdaINTest


parser = argparse.ArgumentParser()
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('--checkpoint', type=str, help='Checkpoint directory', required=True)
parser.add_argument('--style-path', type=str,
                    dest='style_path', help='Style images folder')
parser.add_argument('--video', type=str, help="Stream from input video file", default=None)
parser.add_argument('--video-out', type=str, help="Save to output video file", default=None)
parser.add_argument('--fps', type=int, help="Frames Per Second for output video file", default=10)
parser.add_argument('--no-gui', action='store_true', help="Don't render the gui", default=False)
parser.add_argument('--scale', type=float, help="Scale the output image", default=1)
# parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the style image", default=False)
parser.add_argument('--device', type=str,
                        dest='device', help='Device to perform compute on',
                        default='/gpu:0')
parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping 256x256", default=512)
parser.add_argument('--alpha', type=float, help="Alpha blend value", default=1)
parser.add_argument('--small', action='store_true', help="Use small model architecture", default=False)
parser.add_argument('--concat', action='store_true', help="Concatenate style image and stylized output", default=False)
parser.add_argument('--noise', action='store_true', help="Synthesize textures from noise images", default=False)
parser.add_argument('-r', '--random', type=int, help='Load a random img after iterations', default=0)
args = parser.parse_args()


class WebcamVideoStream:
    '''From http://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/'''
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        (self.ret, self.frame) = self.stream.read()


 
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
 
            # otherwise, read the next frame from the stream
            (self.ret, self.frame) = self.stream.read()
 
    def read(self):
        # return the frame most recently read
        return (self.ret, self.frame)
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class StyleWindow(object):
    def __init__(self, style_path, img_size=512, scale=1, alpha=1):
        self.style_imgs = get_files(style_path)

        self.img_size = img_size
        self.crop_size = 256
        self.scale = scale
        self.alpha = alpha

        cv2.namedWindow('Style Controls')
        if len(self.style_imgs) > 1:
            cv2.createTrackbar('index','Style Controls', 0, len(self.style_imgs)-1, self.set_idx)
        cv2.createTrackbar('alpha','Style Controls', 100, 100, self.set_alpha)
        cv2.createTrackbar('size','Style Controls', img_size, 2048, self.set_size)
        cv2.createTrackbar('crop size','Style Controls', 256, 2048, self.set_crop_size)
        cv2.createTrackbar('scale','Style Controls', int(scale*100), 200, self.set_scale)

        self.set_style(random=True)

    def set_style(self, idx=None, random=False):
        if idx is not None:
            self.idx = idx
        if random:
            self.idx = np.random.randint(len(self.style_imgs))

        style_file = self.style_imgs[self.idx]
        self.style_rgb = get_img_crop(style_file, resize=self.img_size, crop=self.crop_size)
        self.show_style()

    def set_idx(self, idx):
        self.set_style(idx)

    def set_size(self, size):
        self.img_size = max(256, size)  # Don't go below 256
        self.set_style()

    def set_crop_size(self, crop_size):
        self.crop_size = min(crop_size, self.img_size)
        self.set_style()

    def set_scale(self, scale):
        self.scale = scale / 100
        
    def set_alpha(self, alpha):
        self.alpha = alpha / 100

    def show_style(self):
        cv2.imshow('Style Controls', cv2.cvtColor(cv2.resize(self.style_rgb, (args.style_size, args.style_size)), cv2.COLOR_RGB2BGR))


def main():
    if args.video is not None:
        cap = WebcamVideoStream(args.video).start()
    else:
        cap = WebcamVideoStream(args.video_source).start()

    _, frame = cap.read()

    # Grab a sample frame to calculate frame size
    frame_resize = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
    img_shape = frame_resize.shape
    
    ada_in = AdaINTest(args.checkpoint, args.device, args.small)
    
    style_window = StyleWindow(args.style_path, args.style_size, args.scale, args.alpha)

    if args.video_out is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if args.concat:
            out_shape = (img_shape[1]+img_shape[0],img_shape[0])
            print(out_shape)
        else:
            out_shape = (img_shape[1],img_shape[0])
        print('Video Out Shape:', out_shape)
        out = cv2.VideoWriter(args.video_out, fourcc, args.fps, out_shape)

    fps = FPS().start()

    count = 0

    while(True):
        ret, frame = cap.read()

        if ret is True:       
            # frame_resize = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
            frame_resize = cv2.resize(frame, None, fx=style_window.scale, fy=style_window.scale)

            if args.noise:
                frame_resize = np.random.randint(0, 256, frame_resize.shape, np.uint8)
                frame_resize = gaussian_filter(frame_resize, sigma=0.5)

            count += 1
            print("Frame:",count,"Orig shape:",frame.shape,"New shape",frame_resize.shape)

            image_rgb = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, we need RGB

            if args.random > 0 and count % args.random == 0:
                style_window.set_style(random=True)

            # if args.keep_colors:
            #     style_window.style_rgb = coral(style_window.style_rgb, image_rgb)

            # Run the frame through the style network
            stylized_rgb = ada_in.predict(image_rgb, style_window.style_rgb, style_window.alpha)

            if args.concat:
                # Resize style img to same height as frame
                style_rgb_resized = cv2.resize(style_window.style_rgb, (stylized_rgb.shape[0], stylized_rgb.shape[0]))
                stylized_rgb = np.hstack([style_rgb_resized, stylized_rgb])
            
            stylized_bgr = cv2.cvtColor(stylized_rgb, cv2.COLOR_RGB2BGR)
                
            if args.video_out is not None:
                stylized_bgr = cv2.resize(stylized_bgr, out_shape)
                out.write(stylized_bgr)

            if args.no_gui is False:
                cv2.imshow('AdaIN Style', stylized_bgr)

            fps.update()

            key = cv2.waitKey(10) 
            if key & 0xFF == ord('r'):
                style_window.set_style(random=True)
            elif key & 0xFF == ord('q'):
                break
        else:
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    cap.stop()
    
    if args.video_out is not None:
        out.release()
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
