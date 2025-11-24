from pipeline import TrackingPipeline
from modules import config

if __name__ == "__main__":
        video_path = "xxxx-YOUR-VIDEO-PATH-xxxx.mp4"
        output_path = "xxxx-YOUR-OUTPUT-PATH-xxxx.mp4"
        pipeline = TrackingPipeline(video_path, output_path)
        pipeline.run()