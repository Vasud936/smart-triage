import time
from src.rppg_pipeline import RPPGPipeline

pipeline = RPPGPipeline()
pipeline.start()

print("Waiting for camera...")
time.sleep(2)
print("Latest frame:", "Captured" if pipeline.latest_frame is not None else "None")
print("HR:", pipeline.get_current_hr())

pipeline.stop()
