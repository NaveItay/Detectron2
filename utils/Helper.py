import cv2
import numpy as np
import json


class Helper:

    # Sky filter with Panoptic Segmentation output ColorMode
    #                             [{h_min},{h_max},{s_min},{s_max},{v_min},{v_max}]
    sky_hsv_color_mask = np.array([102, 107, 116, 151, 186, 205])

    dataset_no_sky_path = "dataset no-sky"

    lower = sky_hsv_color_mask[0::2]
    upper = sky_hsv_color_mask[1::2]

    def output_yolo_Annotations(self, boxes, highest_sky_pixel, image_width, image_height, counter):

        # initilaized structure
        boxes_yolo_structure = []
        obj = []

        # run on the objects and convert it to yolo format
        for box in boxes:
            if len(box) == 5:

                class_id, x_mid, y_mid, w_box, h_box = self.give_me_correct_box(box)
                y_new_mid = y_mid - highest_sky_pixel   # remove offset
                print(class_id, x_mid, y_new_mid, w_box, h_box)
                print(f'w = {image_width} h = {image_height} ')

                if y_new_mid < 0:
                    y_new_mid = 0
                obj.append(class_id)
                obj.append(x_mid / image_width)
                obj.append(y_new_mid / image_height)
                obj.append(w_box / image_width)
                obj.append(h_box / image_height)
                boxes_yolo_structure.append(obj)
                # print(boxes_yolo_structure)
                obj = []
            else:
                print("no vaild object")  # you can change it

        # print the boxes_yolo_structure array to txt file
        with open("./dataset no-sky/" + str(counter) + '.txt', 'w') as filehandle:
            for listitem in boxes_yolo_structure:
                for i in range(0, len(listitem)):
                    filehandle.write('%s' % listitem[i])
                    filehandle.write(' ')

                filehandle.write('\n')

    def give_me_correct_box(self, box):

        class_id, x_left_up, y_up_left, x_right_down, y_down_right = box
        w = x_right_down - x_left_up
        h = y_down_right - y_up_left
        box = self.get_box_center((class_id, x_left_up, y_up_left, w, h))

        return box

    def get_box_center(self, box):
        class_id, x, y, w, h = box
        cx = x + w // 2
        cy = y + h // 2
        return class_id, cx, cy, w, h

    def sky_filter(self, img):

        y_coordinates = []

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, self.lower, self.upper)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, _, _ = cv2.boundingRect(contour)
            y_coordinates.append(y)
            highest_sky_pixel = max(y_coordinates)

        return mask, highest_sky_pixel

    def crop_image(self, img, y_crop):
        crop_img = img[y_crop:, :]
        height, weight, _ = crop_img.shape

        return crop_img, height, weight

    def to_opencv_format(self, img):
        img = img.get_image()[:, :, ::-1]
        return img

    def stackImages(self, scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                    None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale,
                                             scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver

    def convert_objects_to_coco_structure(self, counter, image_wight, image_height):
        write_json_context = dict()
        write_json_context['objects'] = []
        counter_objects = 1
        with open("./dataset no-sky/" + str(counter) + '.txt', 'r') as f1:
            lines1 = f1.readlines()
        for j, label in enumerate(lines1):
            text = label.split()
            class_id, x_yolo, y_yolo, width_yolo, height_yolo = text
            width_coco = float(width_yolo) * image_wight
            height_coco = float(height_yolo) * image_height
            x_coco = float(x_yolo) * image_wight - (width_coco / 2)
            y_coco = float(y_yolo) * image_height - (height_coco / 2)
            object = {}
            object['object' + str(counter_objects)] = class_id
            object['x_coco'] = x_coco
            object['y_coco'] = y_coco
            object['width_coco'] = width_coco
            object['height_coco'] = height_coco
            write_json_context['objects'].append(object)
            counter_objects += 1
        with open("./dataset no-sky/JSON format/" + str(counter) + '.json', 'w') as fw:
            json.dump(write_json_context, fw)


