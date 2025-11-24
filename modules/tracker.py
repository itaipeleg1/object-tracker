'''
tracker.py - Module for object tracking 

The module contains the ObjectTracker class which track
a detected object across video frames 
'''

import cv2
import numpy as np
from . import config



class ObjectTracker:

    def __init__(self):
        '''
        Initializes the ObjectTracker with a specified tracking algorithm and initial bounding box.

        Parameters:
        tracker_algorithm (str): The tracking algorithm to use. Default is 'CSRT'.
        initial_bbox (tuple): The initial bounding box (x, y, width, height)
        '''
        self.is_initialized = False ## Track if the tracker is initialized
        self.current_bbox = None
        self.algorithm = config.CONFIG['tracker']['algorithm']
        self.pst_track_thresgold = config.CONFIG['tracker']['confidence_psr']

    def create_tracker(self, algorithm):
        params = cv2.TrackerCSRT_Params()
        params.psr_threshold = self.pst_track_thresgold

        if algorithm == 'CSRT':
            return cv2.TrackerCSRT_create(params)
        if algorithm == 'KCF':
            return cv2.TrackerKCF_create()
        
    def initialize(self, frame, bbox):
        # Create fresh tracker for clean reinitialization
        self.tracker = self.create_tracker(self.algorithm)  
        success = self.tracker.init(frame, bbox)
        
        self.is_initialized = True
        self.current_bbox = bbox
        return self.is_initialized
    
    def update(self, frame):

        if not self.is_initialized:
            raise Exception("Tracker is not initialized with an initial bounding box.")
        
        success, bbox = self.tracker.update(frame)
        if success:
            self.current_bbox = bbox
            self.is_initialized = True
        else:
            self.current_bbox = None
            self.is_initialized = False 
            ## RESET tracker if lost
        return success, bbox
    
    def get_current_bbox(self):
        return self.current_bbox
    
    def is_tracking(self):
        return self.is_initialized and self.current_bbox is not None
            


