#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys, os, queue, time, threading, random
import cv2, facenet, align.detect_face

from scipy import misc

import tensorflow as tf
import numpy      as np
import skimage.transform as sktransform


# def getFrameLoop(videoCap, frameBuffer):
#     while True:
#         _, frame = videoCap.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frameBuffer.put(frame)


# def useFrameLoop(frameBuffer):
#     while True:
#         frame = frameBuffer.get()
#         frameBuffer.task_done()

def oneLoop(pnet, rnet, onet):
    minsize, threshold, factor = 20, [ 0.6, 0.7, 0.7 ], 0.709
    videoCap = cv2.VideoCapture(0)

    while True:
        _, frame = videoCap.read()
        frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bounding_boxes, _ = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        alignBoxes(bounding_boxes, frame)
        time.sleep(2)

def alignBoxes(boundingBoxes, frame):

    print('alignBoxes called.')

    nrof_faces = boundingBoxes.shape[0]

    if nrof_faces > 0:
        det = boundingBoxes[:,0:4]
        det_arr = []
        frame_size = np.asarray(frame.shape)[0:2]

        for i in range(nrof_faces):
            det_arr.append(np.squeeze(det[i]))

        nrof_successfully_aligned = 0
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-32/2, 0)
            bb[1] = np.maximum(det[1]-32/2, 0)
            bb[2] = np.minimum(det[2]+32/2, frame_size[1])
            bb[3] = np.minimum(det[3]+32/2, frame_size[0])
            cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = sktransform.resize(cropped, (160, 160))
            nrof_successfully_aligned += 1
            filename_base, file_extension = os.path.splitext('out.jpg')
            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
            misc.imsave(output_filename_n, scaled)
    else:
        print('No faces found.')


def main():
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default(): pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    oneLoop(pnet, rnet, onet)

#     getFrameThreadCount = 1
#     useFrameThreadCount = 5
#     getFrameThreads     = []
#     useFrameThreads     = []

#     videoCapture = cv2.VideoCapture(0)
#     frameBuffer  = queue.Queue(useFrameThreadCount)

#     _, frame = videoCapture.read()
#     frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     for x in range(getFrameThreadCount):
#         getFrameThread = threading.Thread(target=getFrameLoop, args=(videoCapture, frameBuffer))
#         getFrameThread.daemon = True
#         getFrameThread.start()
#         getFrameThreads.append(getFrameThread)

#     for x in range(useFrameThreadCount):
#         useFrameThread = threading.Thread(target=useFrameLoop, args=(frameBuffer,))
#         useFrameThread.daemon = True
#         useFrameThread.start()
#         useFrameThreads.append(useFrameThread)

#     for thread in getFrameThreads:
#         thread.join()

#     for thread in useFrameThreads:
#         thread.join()

if __name__ == '__main__':
    main()
