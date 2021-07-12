# Detectron2 Assignment

### Tasks:

1. Data-set
2. Panoptic Segmentation
3. Manipulate the COCO data-set to the new COCO data-set without sky
4. Present the new data-set

<p>
<br />
<br />
</p>

# 
###### 1. Data-set
> - Video from a dashboard camera in San Francisco, California.
> 
> Video resolution: 1280x720. 
> 
> Frame rate: 30 fps.
> 
> ![alt text](/github_images/youtube.png)
> 
> - Dividing the video into frames
> ```
> save_count = 120    # save every 4 secs, 30fps vid
>     while success:
>
>        # Check if frame need to be saved
>        if frame_count % save_count == 0:
>            print("Save frame " + str(frame_count) + " as " + str(image_name) + ".jpg")
>            # Save frame to image file
>            cv2.imwrite(output_dir_path + str(image_name) + '.jpg', image)
>            # Next image name
>           image_name += 1
> ```
>


###### 2. Panoptic Segmentation
> - Images
> 
> ![alt text](/github_images/Panoptic_Segmentation/example1.jpg)
> 
> ![alt text](/github_images/Panoptic_Segmentation/example2.jpg)
> 
> - Video
> 
> 
> 
> 


### References:

- Data-set
  https://www.youtube.com/watch?v=O1g4Kd9irj4&t=41s&ab_channel=DashCamTours
- TheCodingBug Channel
  https://www.youtube.com/channel/UCcNgapXcZkyW10FIOohZ1uA
- BoundingBox Annotations - YOLO and COCO format 
  https://www.youtube.com/watch?v=qESxRSqsAGw&ab_channel=PrabhjotGosal
- Facebook Open Source Channel
  https://www.youtube.com/watch?v=eUSgtfK4ivk&ab_channel=FacebookOpenSource
- API Documentation » detectron2.structures 
  https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Boxes
- Annolid on Detectron2 Tutorial 
  https://colab.research.google.com/drive/1tv2t7AeUYmXjWC6TrgPNNA23j7ph4zF6
  
