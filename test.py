import os
import cv2
import time

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer

#We are using the pre-trained Detectron2 model, as shown below.
print('Load config')
cfg = get_cfg()
cfg.merge_from_file("keypoint_robot.yaml")

print('Load weight')
# load the pre trained model from Detectron2 model zoo
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

print('Load video')
cap = cv2.VideoCapture('video/igus_240_Trim.mp4')
v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE_BW)

while(cap.isOpened()):
    start_time = time.time() # start time of the loop

    ret, im = cap.read()
    if not ret:
        break

    outputs = predictor(im)

    # v = Visualizer(im[:, :, ::-1],
    #                metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
    #                scale=0.8, 
    #                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    # )
               
    visualization = v.draw_instance_predictions(im, outputs["instances"].to("cpu"))

    # cv2.imshow('frame',im)

    cv2.imshow('res',visualization.get_image())
    
    k = cv2.waitKey(1)
    print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
    # if k == 32:
    #     continue
    # elif k == 27:
    #     break

cap.release()
cv2.destroyAllWindows()