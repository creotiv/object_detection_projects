import logging
import logging.handlers
import os
import time
import sys

import cv2
import numpy as np
import skvideo.io
import utils
import matplotlib.pyplot as plt
# without this some strange errors happen
cv2.ocl.setUseOpenCL(False)

# ============================================================================

IMAGE_DIR = "./out"
VIDEO_SOURCE = '../opencv_traffic_counting/input.mp4'
SHAPE = (720, 1280)
AREA_PTS = np.array([[780, 716], [686, 373], [883, 383], [1280, 636], [1280, 720]]) 

from pipeline import (
    PipelineRunner,
    CapacityCounter,
    ContextCsvWriter
)
# ============================================================================


def main():
    log = logging.getLogger("main")

    base = np.zeros(SHAPE + (3,), dtype='uint8')
    area_mask = cv2.fillPoly(base, [AREA_PTS], (255, 255, 255))[:, :, 0]

    pipeline = PipelineRunner(pipeline=[
        CapacityCounter(area_mask=area_mask, save_image=True, image_dir=IMAGE_DIR),
        # saving every 10 seconds
        ContextCsvWriter('./report.csv', start_time=1505494325, fps=1, faster=10, field_names=['capacity'])
    ], log_level=logging.DEBUG)

    # Set up image source
    cap = skvideo.io.vreader(VIDEO_SOURCE)

    frame_number = -1
    st = time.time()
    
    try:
        for frame in cap:
            if not frame.any():
                log.error("Frame capture failed, skipping...")

            frame_number += 1

            pipeline.set_context({
                'frame': frame,
                'frame_number': frame_number,
            })
            context = pipeline.run()

            # skipping 10 seconds
            for i in xrange(240):
                cap.next()
    except Exception as e:
        log.exception(e)
# ============================================================================

if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
