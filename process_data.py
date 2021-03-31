import os
import logging
import cv2
import sys
import time


from typing import Tuple
import numpy as np
import pandas as pd
import sksurgerycore.transforms.matrix as mu
from sksurgeryimage.calibration.point_detector import PointDetector
import sksurgeryimage.calibration.charuco_plus_chessboard_point_detector \
    as charuco_pd
import sksurgeryimage.calibration.dotty_grid_point_detector as dotty_pd
import sksurgerycalibration.video.video_calibration_driver_stereo as sc

logging.basicConfig(level=logging.INFO, filename='calib.log', filemode='w')
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler())

minimum_points = 50
iterative_iterations = 3

def get_calib_data(directory: str, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_image = cv2.imread(
        os.path.join(directory, f'calib.left.images.{idx}.png')
    )

    right_image = cv2.imread(
        os.path.join(directory, f'calib.right.images.{idx}.png')
    )

    chessboard_tracking = np.loadtxt(
        os.path.join(directory, f'calib.calib_obj_tracking.{idx}.txt')
    )

    scope_tracking = np.loadtxt(
        os.path.join(directory, f'calib.device_tracking.{idx}.txt')
    )

    return left_image, right_image, chessboard_tracking, scope_tracking

def calibrate(left_pd: PointDetector, right_pd: PointDetector, calib_dir: str, use_precalib: bool = False):
    LOGGER.info("Starting Calibration")
    calibration_driver = sc.StereoVideoCalibrationDriver(left_pd,
                                                         right_pd,
                                                         minimum_points)

    if use_precalib:
        left_intrinsics = np.loadtxt('support_data/big_dots_precalib/precalibrated_dots.left.intrinsics.txt')
        left_distortion = np.loadtxt('support_data/big_dots_precalib/precalibrated_dots.left.distortion.txt')
        right_intrinsics = np.loadtxt('support_data/big_dots_precalib/precalibrated_dots.right.intrinsics.txt')
        right_distortion = np.loadtxt('support_data/big_dots_precalib/precalibrated_dots.right.distortion.txt')
        l2r = np.loadtxt('support_data/big_dots_precalib/precalibrated_dots.l2r.txt')
        l2r_rmat = l2r[0:3, 0:3]
        l2r_tvec = l2r[0:3, 3]

    LOGGER.info(f"Grabbing Data: {calib_dir}")

    total_frame_grabbing_time = 0
    num_frames = 10
    for i in range(num_frames):
        start = time.time()
        l_img, r_img, chessboard, scope = get_calib_data(calib_dir, i)
        calibration_driver.grab_data(l_img, r_img, scope, chessboard)
        elapsed = time.time() - start
        LOGGER.info(f"Took {elapsed} seconds to grab data")
        total_frame_grabbing_time += elapsed

    mean_frame_grabbing_time = total_frame_grabbing_time / num_frames
    LOGGER.info("Calibrating")
    
    start = time.time()

    

    if use_precalib:
        LOGGER.info("Using precalibration")
        stereo_reproj_err, stereo_recon_err, stereo_params = \
            calibration_driver.calibrate(
                override_left_intrinsics=left_intrinsics,
                override_left_distortion=left_distortion,
                override_right_intrinsics=right_intrinsics,
                override_right_distortion=right_distortion,
                override_l2r_rmat=l2r_rmat,
                override_l2r_tvec=l2r_tvec)

    else:
        stereo_reproj_err, stereo_recon_err, stereo_params = \
            calibration_driver.calibrate()

    LOGGER.info(f"Calibration (not including hand eye) took: {time.time() - start}")
    tracked_reproj_err, tracked_recon_err, stereo_params = \
        calibration_driver.handeye_calibration()

    elapsed_time = time.time() - start
    return stereo_reproj_err, stereo_recon_err, tracked_reproj_err, tracked_recon_err, elapsed_time, mean_frame_grabbing_time

def create_iterative_ref_data(iterative_image, point_detector, pattern: str):
    """
    Internal method to load a reference image, and create reference
    data suitable for an iterative calibration like Datta 2009.

    :param file_name: reference image file name
    :param point_detector: class derived from PointDetector.
    :return: ids, image_points, image_size
    """

    # Get a new dot detector that doesn't have any distortion coeffs
    if pattern == "dots":
        point_detector = get_ref_dot_detector()

    # Run point detector on image.
    ids, _, image_points = \
        point_detector.get_points(iterative_image)

    return ids, image_points, (iterative_image.shape[1],
                               iterative_image.shape[0])

def iterative_calibrate(left_pd: PointDetector, right_pd: PointDetector, calib_dir, iterative_image_file, iterations, pattern: str):
    LOGGER.info("Iterative Calibration")

    iterative_image = cv2.imread(iterative_image_file)
    iterative_ids, iterative_image_points, iterative_image_size = \
        create_iterative_ref_data(iterative_image, left_pd, pattern)

    calibration_driver = sc.StereoVideoCalibrationDriver(left_pd,
                                                         right_pd,
                                                         minimum_points)

    LOGGER.info(f"Grabbing Data: {calib_dir}")

    total_frame_grabbing_time = 0
    num_frames = 10
    for i in range(num_frames):
        start = time.time()
        l_img, r_img, chessboard, scope = get_calib_data(calib_dir, i)
        calibration_driver.grab_data(l_img, r_img, scope, chessboard)
        elapsed = time.time() - start
        LOGGER.info(f"Took {elapsed} seconds to grab data")
        total_frame_grabbing_time += elapsed

    mean_frame_grabbing_time = total_frame_grabbing_time / num_frames
    LOGGER.info("Calibrating")
        

    start = time.time()

    stereo_reproj_err, stereo_recon_err, stereo_params = \
        calibration_driver.iterative_calibration(iterations,
                                                 iterative_ids,
                                                 iterative_image_points,
                                                 iterative_image_size)

    print(stereo_reproj_err)
    print(stereo_recon_err)
    tracked_reproj_err, tracked_recon_err, stereo_params = \
        calibration_driver.handeye_calibration()

    elapsed_time = time.time() - start
    return stereo_reproj_err, stereo_recon_err, tracked_reproj_err, tracked_recon_err, elapsed_time, mean_frame_grabbing_time

def get_point_detector(intrinsic_matrix, distortion_matrix):
    """
    Returns a point detector based on a set of
    camera intrinsics and distortion coefficient parameters.

    :param intrinsic_matrix: [3x3] matrix
    :param distortion_matrix: [1x5] matrix
    :return:
    """
    number_of_dots = [18, 25]
    dot_separation = 3
    fiducial_indexes = [133, 141, 308, 316]
    reference_image_size = [1900, 2600]
    pixels_per_mm = 40

    number_of_points = number_of_dots[0] * number_of_dots[1]
    model_points = np.zeros((number_of_points, 6))
    counter = 0
    for y_index in range(number_of_dots[0]):
        for x_index in range(number_of_dots[1]):
            model_points[counter][0] = counter
            model_points[counter][1] = (x_index + 1) * pixels_per_mm
            model_points[counter][2] = (y_index + 1) * pixels_per_mm
            model_points[counter][3] = x_index * dot_separation
            model_points[counter][4] = y_index * dot_separation
            model_points[counter][5] = 0
            counter = counter + 1

    dot_detector_params = cv2.SimpleBlobDetector_Params()
    dot_detector_params.filterByConvexity = True
    dot_detector_params.filterByInertia = True
    dot_detector_params.filterByCircularity = False
    #dot_detector_params.minCircularity = 0.7
    dot_detector_params.filterByArea = True
    dot_detector_params.minArea = 50
    dot_detector_params.maxArea = 50000

    point_detector = \
        dotty_pd.DottyGridPointDetector(
            model_points,
            fiducial_indexes,
            intrinsic_matrix,
            distortion_matrix,
            reference_image_size=(reference_image_size[1],
                                  reference_image_size[0]),
            threshold_window_size=301,
            threshold_offset=20,
            dot_detector_params=dot_detector_params
            )

    ## Calib params
    # dot_detector_params = cv2.SimpleBlobDetector_Params()
    # dot_detector_params.filterByConvexity = False
    # dot_detector_params.filterByInertia = True
    # dot_detector_params.filterByCircularity = True
    # #dot_detector_params.minCircularity = 0.7
    # dot_detector_params.filterByArea = True
    # dot_detector_params.minArea = 50
    # dot_detector_params.maxArea = 50000

    # point_detector = \
    #     dotty_pd.DottyGridPointDetector(
    #         model_points,
    #         fiducial_indexes,
    #         intrinsic_matrix,
    #         distortion_matrix,
    #         reference_image_size=(reference_image_size[1],
    #                               reference_image_size[0]),
    #         threshold_window_size=151,
    #         threshold_offset=20,
    #         dot_detector_params=dot_detector_params
    #         )

    return point_detector

def get_ref_dot_detector():

    camera_matrix = np.eye(3)
    distortion_coefficients = np.zeros(5)

    ref_point_detector = get_point_detector(camera_matrix,
                                            distortion_coefficients)
    return ref_point_detector

def get_dot_detectors():

    left_intrinsic_matrix = np.loadtxt("support_data/viking_calib_scikit/calib.left.intrinsics.txt")
    left_distortion_matrix = np.loadtxt("support_data/viking_calib_scikit/calib.left.distortion.txt")
    right_intrinsic_matrix = np.loadtxt("support_data/viking_calib_scikit/calib.right.intrinsics.txt")
    right_distortion_matrix = np.loadtxt("support_data/viking_calib_scikit/calib.right.distortion.txt")

    left_point_detector = \
        get_point_detector(left_intrinsic_matrix, left_distortion_matrix)

    right_point_detector = \
        get_point_detector(right_intrinsic_matrix, right_distortion_matrix)

    return left_point_detector, right_point_detector

def get_charuco_detectors():

    reference_image = cv2.imread("support_data/pattern_4x4_19x26_5_4_with_inset_9x14.png")

    number_of_squares = [19, 26]
    square_tag_sizes = [5, 4]
    filter_markers = True
    number_of_chessboard_squares = [9, 14]
    chessboard_square_size = 3
    chessboard_id_offset = 500

    left_point_detector = \
        charuco_pd.CharucoPlusChessboardPointDetector(
        reference_image,
        minimum_number_of_points=minimum_points,
        number_of_charuco_squares=number_of_squares,
        size_of_charuco_squares=square_tag_sizes,
        charuco_filtering=filter_markers,
        number_of_chessboard_squares=number_of_chessboard_squares,
        chessboard_square_size=chessboard_square_size,
        chessboard_id_offset=500
    )

    right_point_detector = \
        charuco_pd.CharucoPlusChessboardPointDetector(
        reference_image,
        minimum_number_of_points=minimum_points,
        number_of_charuco_squares=number_of_squares,
        size_of_charuco_squares=square_tag_sizes,
        charuco_filtering=filter_markers,
        number_of_chessboard_squares=number_of_chessboard_squares,
        chessboard_square_size=chessboard_square_size,
        chessboard_id_offset=500

    )

    return left_point_detector, right_point_detector

def process_dataset(calib_dir: str, pattern: str, iterative: bool, precalib: bool):

    calibrations = os.listdir(calib_dir)
    n_calibrations = len(calibrations)

    calibrations = os.listdir(calib_dir)

    stereo_reproj_errors = []
    stereo_recon_errors = []
    tracked_reproj_errors = []
    tracked_recon_errors = []
    calib_times = []
    frame_grab_times = []
    successful_dirs = []
    for calibration in calibrations:

        try:
            full_path = os.path.join(calib_dir, calibration)

            if pattern == "charuco":
                left_point_detector, right_point_detector = get_charuco_detectors()
                iterative_image_file = "support_data/pattern_4x4_19x26_5_4_with_inset_9x14.png"
            
            if pattern == "dots":
                left_point_detector, right_point_detector = get_dot_detectors()
                iterative_image_file = "support_data/circles-25x18-r40-s3.png"

            if iterative:
                stereo_reproj_err, stereo_recon_err, tracked_reproj_err, tracked_recon_err, elapsed_time, mean_frame_grabbing_time = \
                    iterative_calibrate(left_point_detector, right_point_detector, full_path, iterative_image_file, iterative_iterations, pattern)
            else:
                stereo_reproj_err, stereo_recon_err, tracked_reproj_err, tracked_recon_err, elapsed_time, mean_frame_grabbing_time = \
                    calibrate(left_point_detector, right_point_detector, full_path, precalib)

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

    labels = ["Reproj.", "Recon.", "Tracked Reproj.", "Tracked Recon.", "Mean Frame Grab Time", "Mean Calibration Time"]
    df = pd.DataFrame([stereo_reproj_errors,
                    stereo_recon_errors,
                    tracked_reproj_errors,
                    tracked_recon_errors,
                    frame_grab_times,
                    calib_times
                    ],
                    index=labels,
                columns=successful_dirs).transpose()

    return df

if __name__ == "__main__":
    charuco_rig = {"dir": "charuco_rig", "pattern": "charuco"}
    dots_rig = {"dir": "dots_rig", "pattern": "dots"}
    dots_user_1 = {"dir": "dots_user_1", "pattern": "dots"}
    dots_user_2 = {"dir": "dots_user_2", "pattern": "dots"}
    charuco_freehand = {"dir": "charuco_freehand", "pattern": "charuco"}
    precalib = {"dir": "Data_orig_folder_structure/calibration_study/precalibration_base_data", "pattern": "charuco"}

    args = sys.argv[1:]
    for dataset in [dots_rig, charuco_rig, charuco_freehand]:

        directory = dataset["dir"]
        pattern = dataset["pattern"]
        
        if 'calib' in args:
            LOGGER.info(f'Processing dataset, non-iterative: {dataset}')
            df = process_dataset(dataset["dir"], dataset["pattern"], iterative=False, precalib=False)
            filename = f'{directory}.csv'
            df.to_csv(filename)

            LOGGER.info(df)

        if 'precalib' in args:
            LOGGER.info(f'Processing dataset, non-iterative, with precalibration: {dataset}')
            df = process_dataset(dataset["dir"], dataset["pattern"], iterative=False, precalib=True)
            filename = f'{directory}-precalib.csv'
            df.to_csv(filename)

            LOGGER.info(df)

        if 'iterative' in args:
            LOGGER.info(f'Processing dataset, with iterative calibration: {dataset}')
            df = process_dataset(dataset["dir"], dataset["pattern"], iterative=True, precalib=False)
            filename = f'{directory}-iterative.csv'
            df.to_csv(filename)
            LOGGER.info(df)
