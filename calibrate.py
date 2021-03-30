import logging
import os
import sys
import pandas as pd
from utils import calibrate, iterative_calibrate, \
    get_dot_detectors, get_charuco_detectors

logging.basicConfig(level=logging.INFO, filename='calib.log', filemode='w')
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler())

# Folder where calibraiton data is
data_dir = sys.argv[1]

is_iterative = False
if "iterative" in sys.argv[1:]:
    is_iterative = True
    iterative_iterations = 3


charuco_rig = {"dir": "charuco_rig", "pattern": "charuco"}
dots_rig = {"dir": "dots_rig", "pattern": "dots"}
charuco_freehand = {"dir": "charuco_freehand", "pattern": "charuco"}

for dataset in [dots_rig, charuco_rig, charuco_freehand]:

    directory = os.path.join(data_dir, dataset["dir"])
    pattern = dataset["pattern"]
    
    LOGGER.info(f'Processing dataset: {dataset}')

    calibrations = os.listdir(directory)

    stereo_reproj_errors = []
    stereo_recon_errors = []
    tracked_reproj_errors = []
    tracked_recon_errors = []
    calib_times = []
    frame_grab_times = []
    successful_dirs = []

    for calibration in calibrations:

        if pattern == "charuco":
            left_point_detector, right_point_detector = get_charuco_detectors()
            iterative_image_file = \
                "support_data/pattern_4x4_19x26_5_4_with_inset_9x14.png"
        
        if pattern == "dots":
            left_point_detector, right_point_detector = \
                get_dot_detectors(is_iterative)
            iterative_image_file = "support_data/circles-25x18-r40-s3.png"

        try:
            full_path = os.path.join(directory, calibration)

            if is_iterative:
                stereo_reproj_err, stereo_recon_err, tracked_reproj_err, \
                  tracked_recon_err, elapsed_time, mean_frame_grabbing_time = \
                    iterative_calibrate(left_point_detector,
                                        right_point_detector,
                                        full_path,
                                        iterative_image_file,
                                        iterative_iterations,
                                        pattern)
            else:
               stereo_reproj_err, stereo_recon_err, tracked_reproj_err, \
                  tracked_recon_err, elapsed_time, mean_frame_grabbing_time = \
                    calibrate(left_point_detector,
                              right_point_detector,
                              full_path)

            LOGGER.info(f"Stereo Reprojection Error: {stereo_reproj_err}")
            LOGGER.info(f"Stereo Reconstruction Error: {stereo_recon_err}")
            LOGGER.info(f"Tracked Reprojection Error: {tracked_reproj_err}")
            LOGGER.info(f"Tracked Reconstruction Error: {tracked_recon_err}")
            LOGGER.info(f"Calibration took: {elapsed_time} seconds")

            stereo_reproj_errors.append(stereo_reproj_err)
            stereo_recon_errors.append(stereo_recon_err)
            tracked_reproj_errors.append(tracked_reproj_err)
            tracked_recon_errors.append(tracked_recon_err)
            calib_times.append(elapsed_time)
            frame_grab_times.append(mean_frame_grabbing_time)

            successful_dirs.append(calibration)

        except Exception as ex:
            LOGGER.error(f"Error processing {full_path}")
            LOGGER.error(ex)

    labels = ["Reproj.", "Recon.", "Tracked Reproj.", "Tracked Recon.", \
              "Mean Frame Grab Time", "Mean Calibration Time"]

    df = pd.DataFrame([stereo_reproj_errors,
                    stereo_recon_errors,
                    tracked_reproj_errors,
                    tracked_recon_errors,
                    frame_grab_times,
                    calib_times
                    ],
                    index=labels,
                columns=successful_dirs).transpose()

    filename = f'{directory}.csv'

    if is_iterative:
        filename = f'{directory}-iterative.csv'

    df.to_csv(filename)

    LOGGER.info(df)