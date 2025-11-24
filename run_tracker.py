from pipeline import TrackingPipeline
from modules import config

if __name__ == "__main__":
    object_rois = config.CONFIG['object_rois']
    #video_path = r"C:\kela\tracker\object-tracker\data\input\MAX_0007.mp4"  # Path to input video
    for video_id in ['MAX_0012','MAX_00051','MAX_0008']:
        if video_id == 'MAX_0008' or video_id == 'MAX_0010':
            config.CONFIG['redetector']['scales']=[0.3,0.5,1.0,1.5,2.2]
        else:
            config.CONFIG['redetector']['scales']=[0.3,0.5,0.8 ,1.0, 1.5]
        if video_id == "MAX_0008":
                
                config.CONFIG['pipeline']['input_resize_factor'] = 1.0
        video_path = fr"C:\kela\tracker\object-tracker\data\input\{video_id}.mp4"  # Path to input video
        output_path = fr"C:\kela\tracker\object-tracker\data\output\{video_id}_tracked_HD7.mp4" # Path to save output video (optional)


        pipeline = TrackingPipeline(video_path, output_path, bbox=object_rois[video_id])
        pipeline.run()