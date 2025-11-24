"""
motion_analyzer.py - Module that runs at the beginning of the tracking pipeline to analyze motion patterns and video segments 
"""

import cv2 as cv
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
from . import config
import os
from tqdm import tqdm

class MotionAnalyzer:
    def __init__(self,path):
        '''
        Initialize the MotionAnalyzer with optical flow parameters and feature extractor for segmentation
        '''
        self.path_video = path
        
        # Load DINOv2 model
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.model.eval()
        self.patch_size = self.model.config.patch_size
        
        # Cache for computed values
        self._motion_speeds = None
        self._segments = None
        self._features = None
        self.still_threshold = config.CONFIG['motion_analyzer']['still_threshold']
        
    def compute_speed_list(self): 
        '''
        Computes average motion speed for each frame in the video using optical flow.
        Returns a list of average speeds per frame.
        '''
        # Return cached result if available
        if self._motion_speeds is not None:
            return self._motion_speeds
        
        cap = cv.VideoCapture(self.path_video) 
        
        ret, prev_frame = cap.read()
        if not ret:
            raise Exception("Failed to read video")
        
        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        first_frame = prev_gray.copy()
        motion_speeds = []
        ## RESIZE FOR FASTER COMPUTATION
        resize_factor = config.CONFIG['motion_analyzer']['optical_flow_resize_factor']
        prev_gray = cv.resize(prev_gray, (int(first_frame.shape[1]*resize_factor), int(first_frame.shape[0]*resize_factor)))
        print("Computing motion speeds for each frame...\n")
        
        for _ in tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)) - 1)):
            
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.resize(frame, (int(first_frame.shape[1]*resize_factor), int(first_frame.shape[0]*resize_factor)))
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            avg_speed = np.mean(mag)
            motion_speeds.append(avg_speed)
            prev_gray = gray
        
        cap.release()
        
        # Cache result
        self._motion_speeds = motion_speeds
        # Cache result
        np.save(f'C:\\kela\\tracker\\object-tracker\\cache\\motion_speeds{os.path.basename(self.path_video)}.npy', np.array(motion_speeds))
        return motion_speeds
    
    def segment_video_by_motion(self):
        '''
        Segments the video into still segments based on motion speeds.
        Returns a list of (start_frame, end_frame) tuples.
        '''
        # Return cached result if available
        if self._segments is not None:
            return self._segments
        if os.path.isfile(f'C:\\kela\\tracker\\object-tracker\\cache\\motion_speeds{os.path.basename(self.path_video)}.npy'):
            motion_speeds = np.load(f'C:\\kela\\tracker\\object-tracker\\cache\\motion_speeds{os.path.basename(self.path_video)}.npy').tolist()
        else:
            motion_speeds = self.compute_speed_list()  # Use cached or compute once
        
        segments = []
        in_segment = False
        start_idx = None
        min_segment_length = config.CONFIG['motion_analyzer']['min_frame_segment_length']
        still_threshold = config.CONFIG['motion_analyzer']['still_threshold']

        for i, speed in enumerate(motion_speeds):
            if speed < still_threshold and not in_segment:
                start_idx = i
                in_segment = True
            elif speed >= still_threshold and in_segment:
                if i - start_idx >= min_segment_length:
                    segments.append((start_idx, i))
                in_segment = False
                start_idx = None
        
        
        # Cache result
        self._segments = segments
        return segments
    
    def features_for_segments(self):
        '''
        Extracts feature vectors for each video segment using pre-trained DINOv2.
        Returns a matrix of shape (n_segments, feature_dim).
        '''
        # Return cached result if available
        if self._features is not None:
            return self._features
        if os.path.isfile(f'C:\\kela\\tracker\\object-tracker\\cache\\motion_speeds{os.path.basename(self.path_video)}.npy'):
            motion_speeds = np.load(f'C:\\kela\\tracker\\object-tracker\\cache\\motion_speeds{os.path.basename(self.path_video)}.npy').tolist()
        else:
            motion_speeds = self.compute_speed_list() 
        segments = self.segment_video_by_motion()  # Get cached segments
        
        
        cap = cv.VideoCapture(self.path_video)  # Open fresh capture
        
        segments_features_mat = []
        

        for i, (start, end) in tqdm(enumerate(segments)):

            
            # Choose the 5 frames with lowest motion
            segment_motions = motion_speeds[start:end]
            
            lowest_motion_indices = np.argsort(segment_motions)[:max(5, int(len(segment_motions)/10))] ## select lowest 10% frames or at least 5 frames
            
            features_list = []
            
            for frame_id in lowest_motion_indices:
                # Read frame
                cap.set(cv.CAP_PROP_POS_FRAMES, start + frame_id)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"  Warning: Could not read frame {start + frame_id}")
                    continue
                
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                inputs = self.processor(images=image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                last_hidden_states = outputs.last_hidden_state
                cls_token = last_hidden_states[:, 0, :]
                features_list.append(cls_token)
            
            if len(features_list) > 0:
                segment_features = torch.cat(features_list, dim=0).mean(dim=0, keepdim=True)
                segments_features_mat.append(segment_features)
        
        cap.release()
        
        if len(segments_features_mat) == 0:
            raise ValueError("No valid segments found")
        
        segments_features_mat = torch.cat(segments_features_mat, dim=0)
        
        # Cache result
        self._features = segments_features_mat
        torch.save(segments_features_mat, f'C:\\kela\\tracker\\object-tracker\\cache\\segment_features{os.path.basename(self.path_video)}_still_threshold{self.still_threshold}.pt')
        return segments_features_mat
    
    def cosine_for_segments(self):
        '''
        Computes cosine similarity matrix between video segments based on their feature vectors.
        Returns a similarity matrix of shape (n_segments, n_segments).
        '''
        if os.path.isfile(f'C:\\kela\\tracker\\object-tracker\\cache\\segment_features{os.path.basename(self.path_video)}_still_threshold{self.still_threshold}.pt'):
            segments_features_mat = torch.load(f'C:\\kela\\tracker\\object-tracker\\cache\\segment_features{os.path.basename(self.path_video)}_still_threshold{self.still_threshold}.pt')
            self._features = segments_features_mat
            # Normalize feature 
            normalized_features = segments_features_mat / segments_features_mat.norm(dim=1, keepdim=True)
            # Compute cosine similarity matrix
            similarity_matrix = torch.matmul(normalized_features, normalized_features.T)
            return similarity_matrix
        else:
            segments_features_mat = self.features_for_segments()  
            # Normalize feature 
            normalized_features = segments_features_mat / segments_features_mat.norm(dim=1, keepdim=True)
            # Compute cosine similarity matrix
            similarity_matrix = torch.matmul(normalized_features, normalized_features.T)
            
            return similarity_matrix