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

minimum_points = 50

LOGGER = logging.getLogger(__name__)

def get_calib_data(directory: str, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Generate correct file names for left/right images and tracking data. """
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

def calibrate(left_pd: PointDetector, right_pd: PointDetector, calib_dir: str,
              stereo_params = None):
    """ Wrapper around calibration algorithm from scikit-surgerycalibration. """
    LOGGER.info("Starting Calibration")
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
    
    if not stereo_params:
        stereo_reproj_err, stereo_recon_err, stereo_params = \
            calibration_driver.calibrate()

    else:
        left_intrinsics = stereo_params.left_params.camera_matrix
        left_distortion = stereo_params.left_params.dist_coeffs
        right_intrinsics = stereo_params.right_params.camera_matrix
        right_distortion = stereo_params.right_params.dist_coeffs
        l2r_rmat = stereo_params.l2r_rmat
        l2r_tvec = stereo_params.l2r_tvec
        
        LOGGER.info("Using precalibration")
        stereo_reproj_err, stereo_recon_err, stereo_params = \
            calibration_driver.calibrate(
                override_left_intrinsics=left_intrinsics,
                override_left_distortion=left_distortion, 
                override_right_intrinsics=right_intrinsics,
                override_right_distortion=right_distortion,
                override_l2r_rmat=l2r_rmat,
                override_l2r_tvec=l2r_tvec)

    LOGGER.info(f"Calibration (not including hand eye) took: {time.time() - start}")
    tracked_reproj_err, tracked_recon_err, stereo_params = \
        calibration_driver.handeye_calibration()

    elapsed_time = time.time() - start
    return stereo_reproj_err, stereo_recon_err, tracked_reproj_err, \
        tracked_recon_err, elapsed_time, mean_frame_grabbing_time, stereo_params

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

def iterative_calibrate(left_pd: PointDetector,
                        right_pd: PointDetector,
                        calib_dir: str,
                        iterative_image_file: str,
                        iterations: int,
                        pattern: str):
    """ Wrapper around iterative calibration algorithm from 
    scikit-surgerycalibration. """

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

    tracked_reproj_err, tracked_recon_err, stereo_params = \
        calibration_driver.handeye_calibration()

    elapsed_time = time.time() - start
    return stereo_reproj_err, stereo_recon_err, tracked_reproj_err, \
        tracked_recon_err, elapsed_time, mean_frame_grabbing_time

def get_dot_params(iterative: bool):
    """ Generate dot detector parameters for calibration. Different parameters
    are used for iterative and non-iterative methods. """
    dot_detector_params = cv2.SimpleBlobDetector_Params()
    dot_detector_params.filterByInertia = True
    dot_detector_params.filterByArea = True
    dot_detector_params.minArea = 50
    dot_detector_params.maxArea = 50000

    if iterative:
        dot_detector_params.filterByConvexity = True
        dot_detector_params.filterByCircularity = False

    else:
        dot_detector_params.filterByConvexity = False
        dot_detector_params.filterByCircularity = True

    return dot_detector_params

def get_point_detector(intrinsic_matrix, distortion_matrix, is_iterative: bool):
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

    dot_detector_params = get_dot_params(is_iterative)

    threshold_offset = 20
    
    threshold_window_size = 151
    if is_iterative:
        threshold_window_size = 301

    point_detector = \
        dotty_pd.DottyGridPointDetector(
            model_points,
            fiducial_indexes,
            intrinsic_matrix,
            distortion_matrix,
            reference_image_size=(reference_image_size[1],
                                  reference_image_size[0]),
            threshold_window_size=threshold_window_size,
            threshold_offset=threshold_offset,
            dot_detector_params=dot_detector_params
            )

    return point_detector

def get_ref_dot_detector():
    """ Return a reference dot detector, for iterative calibration """
    camera_matrix = np.eye(3)
    distortion_coefficients = np.zeros(5)

    ref_point_detector = get_point_detector(camera_matrix,
                                            distortion_coefficients, False)
    return ref_point_detector

def get_dot_detectors(is_iterative: bool):
    """ Return left/right dot detectors. """
    left_intrinsic_matrix = \
        np.loadtxt("support_data/viking_calib_scikit/calib.left.intrinsics.txt")
    left_distortion_matrix = \
        np.loadtxt("support_data/viking_calib_scikit/calib.left.distortion.txt")
    right_intrinsic_matrix = \
        np.loadtxt("support_data/viking_calib_scikit/calib.right.intrinsics.txt")
    right_distortion_matrix = \
        np.loadtxt("support_data/viking_calib_scikit/calib.right.distortion.txt")

    left_point_detector = \
        get_point_detector(left_intrinsic_matrix, left_distortion_matrix, is_iterative)

    right_point_detector = \
        get_point_detector(right_intrinsic_matrix, right_distortion_matrix, is_iterative)

    return left_point_detector, right_point_detector

def get_charuco_detectors():
    """ Return a charuco detector based on the pattern used for calibration. """
    reference_image = cv2.imread("support_data/pattern_4x4_19x26_5_4_with_inset_9x14.png")

    number_of_squares = [19, 26]
    square_tag_sizes = [5, 4]
    filter_markers = True
    number_of_chessboard_squares = [11, 16]
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

def get_detectors(pattern: str, is_iterative: bool):
    """ Return the correct detectors and iterative file for the pattern"""
    if pattern == "charuco":
        left_point_detector, right_point_detector = get_charuco_detectors()
        iterative_image_file = \
            "support_data/pattern_4x4_19x26_5_4_with_inset_9x14.png"
    
    if pattern == "dots":
        left_point_detector, right_point_detector = \
            get_dot_detectors(is_iterative)
        iterative_image_file = "support_data/circles-25x18-r40-s3.png"

    return left_point_detector, right_point_detector, iterative_image_file