

# ============================================================================
# MAIN CONFIGURATION DICTIONARY
# ============================================================================

CONFIG = {

    # ========================================================================
    # INITIAL DETECTION SETTINGS
    # ========================================================================
    'object_embedder': {
        'mode': 'manual',  # 'manual' or 'auto'
    },
    
    # ========================================================================
    # TRACKER SETTINGS
    # ========================================================================
    'tracker': {
        'algorithm': 'CSRT',
        'confidence_psr': 0.05,   #  0.05 is good for most cases  a bit more strict  than default opencv
    },
    
    # ========================================================================
    # VGG RE-DETECTOR SETTINGS
    # ========================================================================
    'redetector': {
        # Model settings
        'device': 'cpu',  # 'cuda' or 'cpu'
        'vgg_layer':30,  # Which VGG layer to use (23=conv4_3, 30=conv5_3, conv3_3=16)
        'stride':16,  # Feature map stride (8 for conv3_3,  16 for conv4_3)
        
        # Confidence thresholds
        'confidence_threshold': 13,  # Minimum PSR for valid detection

        
        # Multi-scale search
        'scales': [0.3,0.5,0.8 ,1.0, 1.5],  # Scales to search over prone to big when object size relative big and vice versa
        
        # Rotation search (add 90, 180, 270 if object can rotate)
        'rotation_angles': [0,90,180,270],
        
        # Matching method
        'method': 'ncc',  # 'ncc' (Normalized Cross-Correlation) or 'cosine'
        
        # Template constraints
        'min_template_size': 4,  # Minimum template size in feature space

    },
        # ========================================================================
    # MOTION ANALYZER SETTINGS
    # ========================================================================
    'motion_analyzer': {
        # Model settings
        'still_threshold': 1,  # Threshold to classify as still
        'min_frame_segment_length': 30,  # Minimum length of still segments in frames
        'optical_flow_resize_factor': 0.1,  # Resize factor for optical flow computation

    },

    # ========================================================================
    # TRACKING PIPELINE SETTINGS
    # ========================================================================
    'pipeline': {
        'visualize_size': 0.3,  # Frames between template updates
        'min_cosine_similarity_for_update': 0.6,  # Minimum cosine similarity to update template
        'lost_frame_redetect_threshold': 30,  # Frames lost before re-detection attempt
        'template_update_interval': 90,  # Frames between template updates
        'cosine_similarity_drift_threshold': 0.6,  # Threshold to detect drift
        'input_resize_factor': 0.5,  # Resize factor for input video frames assign 0.5 for 4k and 1 for lower resolution
        'min_bbox_size': 50,  # Minimum bbox width/height for tracking and redetection (in pixels)

    },
    'object_rois':{
        'MAX_0010':(1920//2, 1560//2, 190//2, 190//2),
        'MAX_0004':(1800//2, 1246//2, 340//2, 410//2),
        'MAX_0007':(1740//2, 1643//2, 450//2, 500//2),
        'MAX_0008':(1940//2, 1693//2, 266//2, 270//2),
        'MAX_0011':(1740//2, 1350//2, 680//2, 696//2),
        'MAX_0012':(1796//2, 850//2, 890//2, 1263//2),
        
        'MAX_00051':(986//2, 886//2, 1156//2, 1610//2)
    }
    
    

    
    
}

