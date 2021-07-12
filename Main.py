from utils.Detector import *
from utils.Helper import Helper
import glob
import cv2
import os

# Input path
dataset_path = glob.glob("dataset/*.jpg")

# Output paths
sky_noSKY_compare_path = "github_images/compare"
dataset_no_sky_path = "dataset no-sky"

img_name_counter = 0

# Init methods
PS_detector = Detector(model_type="PS")  # model_types = OD/IS/KP/LVIS/PS
IS_detector = Detector(model_type="IS")
helper = Helper()

# Image
for path in dataset_path:

    # detect sky with Panoptic Segmentation
    img_detect_sky = PS_detector.onImage(path)

    # convert to openCV format
    img_openCV_format = helper.to_opencv_format(img_detect_sky)

    # sky filter by Panoptic Segmentation ColorMode
    mask, highest_sky_pixel = helper.sky_filter(img_openCV_format)
    print(f'highest_sky_pixel = {highest_sky_pixel}')

    # instance segmentation
    img_segmentation, model_boxes = IS_detector.onImage(path)
    IS_cv_format = helper.to_opencv_format(img_segmentation)

    # remove sky
    img_crop, height_crop, weight_crop = helper.crop_image(IS_cv_format, highest_sky_pixel)

    # output new annotations (YOLO format - txt)
    helper.output_yolo_Annotations(model_boxes, highest_sky_pixel, weight_crop, height_crop, img_name_counter)

    # output annotations (COCO format - JSON)
    helper.convert_objects_to_coco_structure(str(img_name_counter), weight_crop, height_crop)

    # save new dataset - (no sky)
    cv2.imwrite(os.path.join(dataset_no_sky_path, str(img_name_counter) + '.jpg'), img_crop)

    # for GitHub export
    # compare = helper.stackImages(0.5, ([IS_cv_format, ], [img_crop, ]))
    # cv2.imwrite(os.path.join(sky_noSKY_compare_path, str(img_name_counter) + '.jpg'), compare)

    img_name_counter += 1

    # Show result
    # cv2.imshow("Result", compare)
    # cv2.waitKey(0)

# Video
# PS_detector.onVideo("Video for data set/San Francisco_California.mp4")
