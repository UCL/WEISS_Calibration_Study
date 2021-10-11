import logging
import os
import sys
import random
import pandas as pd
from utils import calibrate, iterative_calibrate, \
    get_dot_detectors, get_charuco_detectors

logging.basicConfig(level=logging.INFO, filename='calib.log', filemode='w')
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler())

# Folder where calibraiton data is
data_dir = sys.argv[1]

charuco_rig = {"dir": "charuco_rig", "pattern": "charuco"}
dots_rig = {"dir": "dots_rig", "pattern": "dots"}
charuco_freehand = {"dir": "charuco_freehand", "pattern": "charuco"}

#TODO
is_iterative = False
for dataset in [dots_rig, charuco_rig, charuco_freehand]:

    directory = os.path.join(data_dir, dataset["dir"])

    if not os.path.exists(directory):
        LOGGER.info("Directory doesn't exist: %s, skipping", directory)
        continue

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

    # Choose one dataset at random to use for precalibration

    random.shuffle(calibrations)
    precalibration_data = calibrations[0]
    test_data = calibrations[1:]

    LOGGER.info(f"Using {precalibration_data} for precalibration")

    
    if pattern == "charuco":
        left_point_detector, right_point_detector = get_charuco_detectors()
        iterative_image_file = \
            "support_data/pattern_4x4_19x26_5_4_with_inset_9x14.png"
    
    if pattern == "dots":
        left_point_detector, right_point_detector = \
            get_dot_detectors(is_iterative)
        iterative_image_file = "support_data/circles-25x18-r40-s3.png"

        full_path = os.path.join(directory, precalibration_data)

        stereo_reproj_err, stereo_recon_err, tracked_reproj_err, \
            tracked_recon_err, elapsed_time, mean_frame_grabbing_time, \
                stereo_params = calibrate(left_point_detector,
                                          right_point_detector,
                                          full_path)

        LOGGER.info(f"Stereo Reprojection Error: {stereo_reproj_err}")
        LOGGER.info(f"Stereo Reconstruction Error: {stereo_recon_err}")
        LOGGER.info(f"Tracked Reprojection Error: {tracked_reproj_err}")
        LOGGER.info(f"Tracked Reconstruction Error: {tracked_recon_err}")
        LOGGER.info(f"Calibration took: {elapsed_time} seconds")

    for calibration in test_data:
        LOGGER.info(f"Calibrating {calibration}")
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
            
            stereo_reproj_err, stereo_recon_err, tracked_reproj_err, \
                tracked_recon_err, elapsed_time, mean_frame_grabbing_time, \
                    stereo_params = calibrate(left_point_detector,
                                              right_point_detector,
                                              full_path,
                                              stereo_params)

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

    filename = f'{ dataset["dir"]}.csv'

    if is_iterative:
        filename = f'{ dataset["dir"]}-iterative.csv'

    df.to_csv(filename)

    LOGGER.info(df)

