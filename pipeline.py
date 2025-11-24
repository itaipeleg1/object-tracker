'''
pipeline.py - Main pipeline to run object tracking on a video file and controlling the flow between modules
'''
import cv2
from modules.motion_analyzer import MotionAnalyzer
from modules.roi_selector import ObjectEmbedder
from modules.tracker import ObjectTracker
from modules.vgg_detector import VGGDetector
import os
import numpy as np
from modules.config import CONFIG

class TrackingPipeline:
    def __init__(self, video_path,output_path=None,bbox="manual"):
        self.video_path = video_path
        self.output_path = output_path
        self.visualize_size = CONFIG['pipeline']['visualize_size']
        self.min_cosine_similarity = CONFIG['pipeline']['min_cosine_similarity_for_update']
        self.lost_frame_redetect_threshold = CONFIG['pipeline']['lost_frame_redetect_threshold']
        self.template_update_interval = CONFIG['pipeline']['template_update_interval']
        self.cosine_drift_threshold = CONFIG['pipeline']['cosine_similarity_drift_threshold']
        self.input_resize_factor = CONFIG['pipeline']['input_resize_factor']
        self.min_bbox_size = CONFIG['pipeline']['min_bbox_size']



        ## MODULES
        self.motion_analyzer = MotionAnalyzer(video_path)
        self.tracker = ObjectTracker()
        self.redetector = VGGDetector()
        self.object_embedder = ObjectEmbedder() ## Now only use for manual ROI selection
        

        # Precompute motion analysis
        print("Analyzing video motion and segments...")
        self.motion_analyzer = MotionAnalyzer(video_path)
        self.segments = self.motion_analyzer.segment_video_by_motion()
        self.similarity_matrix = self.motion_analyzer.cosine_for_segments()
        if os.path.isfile(f'C:\\kela\\tracker\\object-tracker\\cache\\motion_speeds{os.path.basename(self.video_path)}.npy'):
            self.motion_speeds = np.load(f'C:\\kela\\tracker\\object-tracker\\cache\\motion_speeds{os.path.basename(self.video_path)}.npy').tolist()
        else:
            self.motion_speeds = self.motion_analyzer.compute_speed_list()

        print(f"Found {len(self.segments)} still segments")


                  # Step 1: ROI selection on first frame
        print("\nSelecting ROI...")
        cap = cv2.VideoCapture(video_path)
        
        ret, first_frame = cap.read()
        if not ret:
              raise Exception("Failed to read the first frame")

          # Resize for ROI selection
        #frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        frame_bgr = first_frame.copy()
        frame_resized = cv2.resize(frame_bgr,
                                     (int(first_frame.shape[1] * self.input_resize_factor),
                                      int(first_frame.shape[0] * self.input_resize_factor)))

          # Select ROI on resized frame
        if bbox == "manual":
            roi_bbox_resized = self.object_embedder.define_roi(frame_resized)  ## (x, y, w, h)
        else:
            roi_bbox_resized = bbox

          # Scale bbox back to original size

      

          # Validate bbox is within frame bounds
        frame_h, frame_w = first_frame.shape[:2]
        x, y, w, h = roi_bbox_resized
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        self.initial_bbox = (x, y, w, h)

        print(f"Frame size: {frame_w}x{frame_h}")
        print(f"Initial bbox (scaled): {roi_bbox_resized}")
        print(f"Initial bbox (clamped): {self.initial_bbox}")

          # Initialize tracker with first frame
        init_success = self.tracker.initialize(frame_resized, self.initial_bbox)
        if not init_success:
              raise Exception(f"Failed to initialize tracker with bbox {self.initial_bbox} on frame size {frame_w}x{frame_h}")
        first_roi = frame_resized[
                self.initial_bbox[1]:self.initial_bbox[1]+self.initial_bbox[3],
                self.initial_bbox[0]:self.initial_bbox[0]+self.initial_bbox[2]
            ]
        self.template_features = self.redetector._extract_features(first_roi, is_template = True)
        self.original_template_features = self.template_features  # Keep backup of original

        cap.release()

          # State
        self.current_bbox = self.initial_bbox
        self.frames_lost = 0
        self.last_good_bbox_size = (self.initial_bbox[2], self.initial_bbox[3])  # Track bbox size
          # Statistics
        self.total_frames_tracked = 0
        self.total_frames_lost = 0
        self.redetection_attempts = 0
        self.redetection_successes = 0

        print("\nPipeline initialized and ready to run!\n")

    def calculate_cosine_similarity(self, feat1, feat2):
        '''
        Calculate cosine similarity between two feature tensors
        '''
        feat1_np = feat1.cpu().numpy().flatten()
        feat2_np = feat2.cpu().numpy().flatten()
        dot_product = np.dot(feat1_np, feat2_np)
        norm_product = np.linalg.norm(feat1_np) * np.linalg.norm(feat2_np)
        cosine_similarity = dot_product / (norm_product + 1e-10)
        return cosine_similarity
    def run(self):
        '''
        Main tracking loop
        '''

        # Initialize video capture
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)* self.input_resize_factor)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)* self.input_resize_factor)

        print("Starting tracking...\n")

        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, 30.0, (width, height))
        
        print("Starting main tracking loop...HAVE FUN\n")

        ## infex
        frame_index = 0
        current_segment = 0
        lost_segment = 0
        redetect_success = False
        tracking_sequence = 0 
        is_lost = False
        bbox_too_small = False
        frames_since_redetect_attempt = 0
        redetect_interval = 30  # Minimum frames between re-detection attempts
        failed_redetections = 0  # Track consecutive failures
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.input_resize_factor != 1.0:
                frame = cv2.resize(frame, None, fx=self.input_resize_factor,
                         fy=self.input_resize_factor)
            ## log speed
            if frame_index < len(self.motion_speeds):
                cv2.putText(frame, f"Motion speed: {self.motion_speeds[frame_index]:.2f}", (10,130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 8)

                
            frame_index += 1
            # Update tracker
            if is_lost == False:
                tracking_success, bbox = self.tracker.update(frame)
            vis_frame = frame.copy()
            for i, s in enumerate(self.segments):
                if s[0] <= frame_index <= s[1]:
                    current_segment = i
                    break

            if is_lost == False and tracking_success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.circle(vis_frame, (x + w // 2, y + h // 2), max(h//2, w//2), (0, 255, 0), 6)
                self.total_frames_tracked += 1
                self.current_bbox = bbox

                # Check if bbox is too big compared to initial bbox
                initial_w, initial_h = self.initial_bbox[2], self.initial_bbox[3]
                size_ratio = max(w / initial_w, h / initial_h)



                bbox_too_small = (w <= self.min_bbox_size or h <= self.min_bbox_size) ## Minimum for vgg redetector
                if self.input_resize_factor >1.0:
                    bbox_too_small =False
                if bbox_too_small:
                    print(f"Warning: BBox too small at frame {frame_index} size ({w}x{h}), enforcing minimum size {self.min_bbox_size}x{self.min_bbox_size}")
                    x, y, w, h = bbox
                    frame_h, frame_w = frame.shape[:2]
                    # Calculate center
                    cx = x + w / 2
                    cy = y + h / 2
                    # Expand to minimum size
                    new_w = max(w, self.min_bbox_size)
                    new_h = max(h, self.min_bbox_size)
                    # Recalculate top-left to keep center
                    x = int(cx - new_w / 2)
                    y = int(cy - new_h / 2)
                    w = int(new_w)
                    h = int(new_h)

                    # Clamp to frame bounds
                    x = max(0, min(x, frame_w - w))
                    y = max(0, min(y, frame_h - h))
                    w = min(w, frame_w - x)
                    h = min(h, frame_h - y)

                    # Update bbox and reinitialize tracker with expanded region
                    bbox = (x, y, w, h)
                    self.current_bbox = bbox
                    self.tracker.initialize(frame, bbox)

                self.frames_lost = 0
                tracking_sequence += 1
                ## Update template periodically with size validation
                if frame_index % self.template_update_interval == 0 and not bbox_too_small:
                        print(f"Periodic Check with initial template {frame_index}")
                        # Extract current tracked region
                        x, y, w, h = [int(v) for v in self.current_bbox]
                        if (x >= 0 and y >= 0 and
                            x + w <= frame.shape[1] and
                            y + h <= frame.shape[0]):

                            current_roi = frame[y:y+h, x:x+w]
                            current_features, _ = self.redetector._extract_features(current_roi, is_template=False)
                        current_descriptor = current_features.mean(dim=[1, 2])  # [C, H, W] -> [C] GLOBAL AVG POOL
                        best_cosine = -1
                        for template_data in self.original_template_features:
                            template_features = template_data['features']
                            template_descriptor = template_features.mean(dim=[1, 2])  # [C, H, W] -> [C]
                            ## compute cosine similarity
                            current_feat_np = current_descriptor.cpu().numpy().flatten()
                            template_feat_np = template_descriptor.cpu().numpy().flatten()
                            dot_product = np.dot(current_feat_np, template_feat_np)
                            norm_product = np.linalg.norm(current_feat_np) * np.linalg.norm(template_feat_np)
                            cosine_similarity = dot_product / (norm_product + 1e-10)
                            best_cosine = max(best_cosine, cosine_similarity)

                        if best_cosine < self.cosine_drift_threshold:
                            print(f"Drift detected Reverting to original template due to low similarity at frame {frame_index} COSINESIM with original  {best_cosine:.3f}")
                            #self.template_features = self.original_template_features
                            is_lost = True  # Force re-detection
                            self.frames_lost = self.lost_frame_redetect_threshold
                            
                        
                        else:
                            x_int, y_int, w_int, h_int = [int(v) for v in self.current_bbox]
                            print(f"Updating template at frame {frame_index}")
                                ## Check if bbox is within frame bounds
                            if (x_int >= 0 and y_int >= 0 and
                                    x_int + w_int <= frame.shape[1] and
                                    y_int + h_int <= frame.shape[0]):

                                    roi = frame[y_int:y_int+h_int, x_int:x_int+w_int]
                                    self.template_features = self.redetector._extract_features(roi, is_template=True)
                        print(f"COSINESIM with original template at frame {frame_index} : {best_cosine:.3f}")
                                
                #if tracking_sequence % 150 ==0:
                 #   print(f"Periodic Check with initial template {frame_index}")
                    # Extract current tracked region
                 ##   x, y, w, h = [int(v) for v in self.current_bbox]
                 #   if (x >= 0 and y >= 0 and
                 #       x + w <= frame.shape[1] and
                 #       y + h <= frame.shape[0]):

                  #      current_roi = frame[y:y+h, x:x+w]
                  #      current_features, _ = self.redetector._extract_features(current_roi, is_template=False)
                   # current_descriptor = current_features.mean(dim=[1, 2])  # [C, H, W] -> [C] GLOBAL AVG POOL
                  #  best_cosine = -1
                 #   for template_data in self.original_template_features:
                 #       template_features = template_data['features']
                  #      template_descriptor = template_features.mean(dim=[1, 2])  # [C, H, W] -> [C]
                        ## compute cosine similarity
                  #      current_feat_np = current_descriptor.cpu().numpy().flatten()
                  #      template_feat_np = template_descriptor.cpu().numpy().flatten()
                  #      dot_product = np.dot(current_feat_np, template_feat_np)
                  #      norm_product = np.linalg.norm(current_feat_np) * np.linalg.norm(template_feat_np)
                  #      cosine_similarity = dot_product / (norm_product + 1e-10)
                  #      best_cosine = max(best_cosine, cosine_similarity)

                   # if best_cosine < self.cosine_drift_threshold:
                   #     print(f"Drift detected Reverting to original template due to low similarity at frame {frame_index} COSINESIM with original  {best_cosine:.3f}")
                   #     self.template_features = self.original_template_features
                    #    is_lost = True  # Force re-detection
                    #    self.frames_lost = self.lost_frame_redetect_threshold
                   # print(f"COSINESIM with original template at frame {frame_index} : {best_cosine:.3f}")

            else:
                is_lost = True
                cv2.putText(vis_frame, "Lost", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 8)

                self.total_frames_lost += 1
                self.frames_lost += 1
                frames_since_redetect_attempt += 1
                lost_speed = self.motion_speeds[frame_index] if frame_index < len(self.motion_speeds) else 0

                ## Attempt redetection only if conditions are met
                should_attempt_redetection = False

                if self.frames_lost >= self.lost_frame_redetect_threshold and frames_since_redetect_attempt >= redetect_interval:
                    # Check if in still segment and scene similarity
                    if lost_speed < CONFIG['motion_analyzer']['still_threshold']:


                        # Get the segment index
                        for i, s in enumerate(self.segments):
                            if s[0] <= frame_index <= s[1]:
                                lost_segment = i
                                break

                        # Check scene similarity
                        if self.similarity_matrix[0, current_segment] < self.min_cosine_similarity:
                            cv2.putText(vis_frame, "Skipping re-detection due to low scene similarity/Speed", (10,180),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 8)

                        else:
                            should_attempt_redetection = True

                    else:
                        # Not in still segment, but still try if enough time has passed
                        cv2.putText(vis_frame, "Attempting re-detection (although not still segment)", (10,150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 8)
                        should_attempt_redetection = True

                # Perform redetection if conditions met
                if should_attempt_redetection:
                    print(f"Re-detection attempt at frame {frame_index} using current template")
                    cv2.putText(vis_frame, f"Attempting re-detection (current)...", (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 8)
                    combined_features = self.template_features + self.original_template_features
                    redetect_success, redetect_bbox, _ = self.redetector.redetect(
                        frame, combined_features)
                    self.redetection_attempts += 1
                    frames_since_redetect_attempt = 0
                    ## check cosine between redetected and original template
                    if redetect_success:
                        failed_redetections = 0  # Reset failure counter



                    #if failed_redetections >=5:
                     #   # After multiple failures, try with original template
                      #  print(f"Re-detection attempt at frame {frame_index} using original template after {failed_redetections} failures")
                     #   cv2.putText(vis_frame, f"Attempting re-detection (original)...", (10,90),
                    #                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 8)
                   #     redetect_success, redetect_bbox, _ = self.redetector.redetect(
                   #         frame, self.original_template_features)
                   #     self.redetection_attempts += 1
                  #      frames_since_redetect_attempt = 0


                
                

                if redetect_success:
                    # Validate and clamp bbox to frame bounds
                    x, y, w, h = redetect_bbox
                    frame_h, frame_w = frame.shape[:2]
                    MIN_BBOX_SIZE = 50  # Minimum size for CSRT to work reliably
                    # Calculate center
                    cx = x + w / 2
                    cy = y + h / 2
                    # Expand to minimum size if needed
                    new_w = max(w, MIN_BBOX_SIZE)
                    new_h = max(h, MIN_BBOX_SIZE)

                    # Recalculate top-left to keep center
                    x = int(cx - new_w / 2)
                    y = int(cy - new_h / 2)
                    w = int(new_w)
                    h = int(new_h)

                    # Make sure bbox is within frame bounds
                    x = max(0, min(x, frame_w - w))
                    y = max(0, min(y, frame_h - h))
                    w = min(w, frame_w - x)
                    h = min(h, frame_h - y)

                    redetect_bbox = (x, y, w, h)
                    cv2.putText(vis_frame, "Re-detection successful", (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 8)
                    if len(frame.shape) < 3:
                            print("Frame shape less than 3 channels, skipping frame")
                            continue

                    self.redetection_successes += 1
                    self.tracker.initialize(frame, redetect_bbox)
                    self.current_bbox = redetect_bbox
                    self.frames_lost = 0
                    is_lost = False  # Resume tracking
                    failed_redetections = 0  # Reset failure counter
                    redetect_success = False  # Reset for next time
                elif should_attempt_redetection:
                    failed_redetections += 1  # Increment failure counter
                    cv2.putText(vis_frame, f"Re-detection failed (attempt {failed_redetections})", (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 8)
                    
                    


            # Visualization
            vis_frame_resized = cv2.resize(vis_frame,
                                           (int(vis_frame.shape[1] * self.visualize_size),
                                            int(vis_frame.shape[0] * self.visualize_size)))
            cv2.imshow("Tracking", vis_frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if self.output_path:
                out.write(vis_frame)
        if self.output_path:
            out.release()
        cv2.destroyAllWindows()
        cap.release()

        ## STATISTICS
        print("\nTracking complete!")
        print(f"Total frames: {total_frames}")
        print(f"Total frames tracked: {self.total_frames_tracked}")
        print(f"Total frames lost: {self.total_frames_lost}")
        print(f"Redetection attempts: {self.redetection_attempts}")
        print(f"Redetection successes: {self.redetection_successes}")