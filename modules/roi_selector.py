'''
roi_selector.py - Module for initial object detection 

'''

import cv2
import numpy as np
from . import config




class ObjectEmbedder:

    def __init__(self):
        '''
        An object embedder using pre-trained VGG16 for feature extraction.
        Used at the initial detection stage.
        '''
        self.current_bbox = None
        self.mode = config.CONFIG['object_embedder']['mode']

    def define_roi(self, frame):
        '''
        A function to define the Region of Interest (ROI) for the object to be tracked.

        Parameters:
        frame (numpy.ndarray): The video frame on which to define the ROI.

        Returns:
        tuple: The bounding box (x, y, width, height) defined by the user.
        '''

        if self.mode == 'manual':
            bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Object to Track")
            self.current_bbox = bbox
            return bbox
        



