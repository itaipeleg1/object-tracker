"""
vgg_detector.py - Module for re-detecting lost objects using VGG 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from . import config
import tqdm

class VGGDetector:
    def __init__(self):
        '''
        Initializes the VGGRedetector with a pre-trained VGG16 model for feature extraction.
        '''

        ## LOAD AND SET VGG MODEL
        self.device = torch.device(config.CONFIG['redetector']['device'])
        self.model = models.vgg16(pretrained=True)
        vgg_layer = config.CONFIG['redetector']['vgg_layer']
        self.feature_extractor = self.model.features[:vgg_layer].to(self.device)
        self.feature_extractor.eval()

        ### LOAD PARAMS FROM CONFIG
        self.psr_confidence_threshold = config.CONFIG['redetector']['confidence_threshold']
        self.scales = config.CONFIG['redetector']['scales']
        self.rotation_angles = config.CONFIG['redetector']['rotation_angles']
        self.sim_method = config.CONFIG['redetector']['method']
        self.min_template_size = config.CONFIG['redetector']['min_template_size']
        self.stride = config.CONFIG['redetector']['stride']  
        
        # Statistics
        self.success_count = 0
        self.attempt_count = 0  
        self.last_known_bbox = None
        
        # Preprocessing transformation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def _extract_features(self, image,is_template=False):
        """Extract VGG spatial features from image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        scales = self.scales if is_template else [1.0]
        rotation_angles = self.rotation_angles if is_template else [0]

        if is_template:
            features_list = []
            with torch.no_grad():
                for scale in scales:
                    for angle in rotation_angles:
                        # Apply scaling and rotation to img_tensor here if needed
                        orig_h, orig_w = image_rgb.shape[:2]
                        scaled_h, scaled_w = int(orig_h * scale), int(orig_w * scale)
                        image_rgb_scaled = cv2.resize(image_rgb, (scaled_w, scaled_h))
                        

                        if angle != 0:
                            rotation_map = {
                                90: cv2.ROTATE_90_CLOCKWISE,
                                180: cv2.ROTATE_180,
                                270: cv2.ROTATE_90_COUNTERCLOCKWISE
                            }
                            image_rgb_rotated = cv2.rotate(image_rgb_scaled, rotation_map[angle])
                        else:
                            image_rgb_rotated = image_rgb_scaled
                    
                        img_tensor_rotated = self.transform(image_rgb_rotated).unsqueeze(0).to(self.device)
                        template_features = self.feature_extractor(img_tensor_rotated).squeeze(0)
                        ## SKIP TOO SMALL TEMPLATES
                                              # Skip if too small
                        if (template_features.shape[1] < self.min_template_size or
                            template_features.shape[2] < self.min_template_size):
                            continue

                        # Store features with metadata
                        features_list.append({
                            'features': template_features,
                            'scale': scale,
                            'rotation': angle,
                            'size': (image_rgb_rotated.shape[1], image_rgb_rotated.shape[0])  # w, h
                        })

            return features_list

        else:
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
        
                return features.squeeze(0), image_rgb.shape[:2]
    
    def spatial_match(self, template_features, search_features): 
        ''' 
        Perform spatial matching between template and search features
        
        Parameters:
        - template_features: [n,C x H_t x W_t] tensor
        - search_features: [C x H_s x W_s] tensor
        '''
        C, H_t, W_t = template_features.shape
        
        if self.sim_method == 'ncc':
            ## NORMALIZED CROSS CORRELATION
            template_kernel = template_features.unsqueeze(0)
            search_input = search_features.unsqueeze(0)

            ones_kernel = torch.ones_like(template_kernel)
            template_mean = template_features.mean(dim=0, keepdim=True)

            search_sum = F.conv2d(search_input, ones_kernel, padding=0)
            search_mean = search_sum / (C * H_t * W_t) 
 
            template_centered = template_features - template_mean
            template_kernel_centered = template_centered.unsqueeze(0)
            
            numerator = F.conv2d(search_input, template_kernel_centered, padding=0)
            
            template_std = template_centered.pow(2).sum().sqrt()
            
            search_sq = search_input.pow(2)
            search_sq_sum = F.conv2d(search_sq, ones_kernel, padding=0)
            search_variance = (search_sq_sum / (C * H_t * W_t)) - search_mean.pow(2)  
            search_std = search_variance.sqrt().clamp(min=1e-8)
            
            similarity_map = numerator / (template_std * search_std + 1e-8)
            similarity_map = similarity_map.squeeze().cpu().numpy()

        elif self.sim_method == 'cosine':
            # Cosine similarity
            template_norm = F.normalize(template_features, dim=0)
            search_norm = F.normalize(search_features, dim=0)
            
            template_kernel = template_norm.unsqueeze(0)
            search_input = search_norm.unsqueeze(0)
            
            similarity_map = F.conv2d(search_input, template_kernel, padding=0)
            similarity_map = similarity_map.squeeze().cpu().numpy()
        
        else:
            raise ValueError(f"Unknown similarity method: {self.sim_method}")
        
        ## DEAL WITH BORDERS
        border_margin = max(2, int(min(H_t, W_t) * 0.5))  
        
        valid_mask = np.ones_like(similarity_map, dtype=bool)
        valid_mask[:border_margin, :] = False
        valid_mask[-border_margin:, :] = False
        valid_mask[:, :border_margin] = False
        valid_mask[:, -border_margin:] = False

        return similarity_map, valid_mask
    
    def find_best_peak(self, similarity_map, valid_mask):
        ''' Find the best peak in the similarity map and compute PSR '''
        masked_map = similarity_map.copy()
        masked_map[~valid_mask] = -np.inf
        
        if np.all(np.isinf(masked_map)):
            return (0, 0), -np.inf
        
        peak_score = masked_map.max()
        best_idx = np.unravel_index(np.argmax(masked_map), masked_map.shape)
        
        # Compute PSR
        valid_scores = similarity_map[valid_mask] 
        if len(valid_scores) > 0:
            mean_score = valid_scores.mean()
            std_score = valid_scores.std()
            psr = (peak_score - mean_score) / (std_score + 1e-8)
        else:
            psr = -np.inf

        return best_idx, psr
    

    def redetect(self, frame, template_features):
        """
        Re-detect the template in a new frame

        Parameters:
        - frame: Full frame to search in (BGR image)
        - template_features: Pre-extracted template features from _extract_features()

        Returns:
        - success: bool (True if confident detection found)
        - bbox: (x, y, w, h) or None
        - best_result: dict with PSR, scale, rotation, bbox
        """
        self.attempt_count += 1
        search_features, _ = self._extract_features(frame)
        template_image_features = template_features

        
        peak_score = -np.inf 
        best_result = {
            'psr': -np.inf, 
            'scale': None, 
            'rotation': None, 
            'bbox': None
        }
        '''
        # Multi-scale and rotation search
       # for scale in self.scales:
        #    for rotation in self.rotation_angles:
         #       # Scale template
          #      orig_h, orig_w = template_image.shape[:2]
          #      scaled_h, scaled_w = int(orig_h * scale), int(orig_w * scale)
                
          #      if scaled_h >= frame.shape[0] or scaled_w >= frame.shape[1]:
           #         continue
                
          #      scaled_template = cv2.resize(template_image, (scaled_w, scaled_h))
                
                # Rotate if needed
           #     if rotation != 0:
          ##          rotation_map = {
             #           90: cv2.ROTATE_90_CLOCKWISE,
            #            180: cv2.ROTATE_180,
              #          270: cv2.ROTATE_90_COUNTERCLOCKWISE
             #       }
              #      rotated_template = cv2.rotate(scaled_template, rotation_map[rotation])
          #      else:
           #         rotated_template = scaled_template
          '''   
        for template_data in tqdm.tqdm(template_image_features):
                template_features = template_data['features']
                scale = template_data['scale']
                rotation = template_data['rotation']
                template_size = template_data['size'] # a tuple (w,h)

                # Skip if template features are larger than search features
                C_t, H_t, W_t = template_features.shape
                C_s, H_s, W_s = search_features.shape

                if H_t > H_s or W_t > W_s:
                    print(f"Skipping template (scale={scale}, rot={rotation}): template ({H_t}x{W_t}) > search ({H_s}x{W_s})")
                    continue

                # Perform matching
                similarity_map, valid_mask = self.spatial_match(
                    template_features, search_features
                )

                peak_position, psr = self.find_best_peak(
                    similarity_map, valid_mask
                )

                if np.isinf(psr):
                    continue
                
                # Convert feature coords to image coords
                peak_y_feat, peak_x_feat = peak_position
                y_img = peak_y_feat * self.stride  
                x_img = peak_x_feat * self.stride
                
                # Update best if PSR is better
                if psr > peak_score:
                    peak_score = psr
                    best_result = {
                        'psr': psr,
                        'scale': scale,
                        'rotation': rotation,
                        'bbox': (x_img, y_img, template_size[0], template_size[1])
                    }
        
        # Check confidence threshold
        success = (peak_score >= self.psr_confidence_threshold)
        
        if success:
            self.success_count += 1
            self.last_known_bbox = best_result['bbox']
            print(f"Re-detected! PSR={best_result['psr']:.2f}, ")

        else:
            print(f"Re-detection failed. PSR={peak_score:.2f} < {self.psr_confidence_threshold}")
        
        return success, best_result['bbox'], best_result
    
    def get_statistics(self):
        """Get re-detection statistics"""
        return {
            'attempts': self.attempt_count,
            'successes': self.success_count,
            'success_rate': self.success_count / max(1, self.attempt_count)
        }