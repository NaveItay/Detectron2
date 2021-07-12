# import some common Detectron2 utilities
# from detectron2.data.catalog import DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo

import cv2


class Detector:

    boxes_numpy = [[]]

    def __init__(self, model_type="OD"):
        self.cfg = get_cfg()
        self.model_type = model_type

        # Load model config and pretrained model
        if model_type == "OD":  # object detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

        elif model_type == "IS":  # instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        elif model_type == "KP":  # keypoint detection
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

        elif model_type == "LVIS":  # LVIS Segmentation
            self.cfg.merge_from_file(
                model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

        elif model_type == "PS":  # Panoptic Segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.DEVICE = "cuda"  # cuda or cpu

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath):

        model_format = []

        image = cv2.imread(imagePath)

        if self.model_type != "PS":

            outputs = self.predictor(image)
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            classes = outputs["instances"].pred_classes
            self.boxes_numpy = outputs["instances"].pred_boxes.tensor.cpu().numpy()

            for count, class_id in enumerate(classes):
                class_id_int = class_id.item()  # class id number
                # index_class = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes[class_id_int]
                yolo_row = self.writeAnnotations(count, class_id_int)
                model_format.append(yolo_row)

            return out, model_format
        else:
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                           instance_mode=ColorMode.SEGMENTATION)  # SEGMENTATION/IMAGE
            out = v.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
            return out

    def writeAnnotations(self, count, class_id_int):
        x, y, w, h = self.boxes_numpy[count]
        return class_id_int, x, y, w, h

    def onVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)

        # video out
        out_video = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (1280, 720))

        while cap.isOpened():
            _, current_frame = cap.read()

            if self.model_type != "PS":

                outputs = self.predictor(current_frame)
                v = Visualizer(current_frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            else:
                predictions, segmentInfo = self.predictor(current_frame)["panoptic_seg"]
                v = Visualizer(current_frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                out = v.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
                out = out.get_image()[:, :, ::-1]

                # Write the frame into the file 'output.avi'
                # out_video.write(out)

            cv2.imshow("Result", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            print("Error opening the video")
            return

        # When everything done, release the video capture and video write objects
        cap.release()
        out.release()
        # Closes all the frames
        cv2.destroyAllWindows()
