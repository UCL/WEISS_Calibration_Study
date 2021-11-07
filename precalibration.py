import logging
import os
import sys
import random
import pandas as pd
from utils import calibrate, get_detectors, iterative_calibrate, \
    get_dot_detectors, get_charuco_detectors, get_detectors

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
    
    # Loop over every set of collected images and use each for precalibration,
    # then calibrate on the rest. This takes a long time!


    left_point_detector, right_point_detector, iterative_image_file = \
        get_detectors(pattern, is_iterative)

    for i, calib_set in enumerate(calibrations):
        stereo_reproj_errors = []
        stereo_recon_errors = []
        tracked_reproj_errors = []
        tracked_recon_errors = []
        calib_times = []
        frame_grab_times = []
        successful_dirs = []

        # Split into precalib set and the rest
        precalibration_data = calibrations[i]
        test_data = (c for j, c in enumerate(calibrations) if j != i)

        LOGGER.info(f"Using {precalibration_data} for precalibration")

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

        output_dir = f'results/precalib/{dataset["dir"]}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = f'{ dataset["dir"]}-precalib-{calibrations[i]}.csv'

        if is_iterative:
            filename = f'{ dataset["dir"]}-precalib-{calibrations[i]}-iterative.csv'

        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path)

        LOGGER.info(df)

