
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
CustomObjectScope = tf.keras.utils.CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from .metrics import dice_coef, dice_loss, iou
import pickle
from PIL import Image,ImageOps

H = 256
W = 256

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return ori_x, x                                ## (1, 256, 256, 3)


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)                    ## (256, 256)
    return ori_x, x

def save_results(y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    # ori_y = np.expand_dims(ori_y, axis=-1)  ## (256, 256, 1)
    # ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)  ## (256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) ## (256, 256, 3)

    cat_images = np.concatenate([line, y_pred*255], axis=1)
    cv2.imwrite(save_image_path, cat_images)

    return cat_images




def predictMask(imageName):
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)


    """ Load the model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
        model = tf.keras.models.load_model("D:/MMST/s2(22)/Machine learning/webApp/static/model/model.h5")

    SCORE = []


    """ Read the image """
    ori_x, x = read_image("D:/MMST/s2(22)/Machine learning/webApp/static/images/"+imageName)

    """ Predicting the mask """
    y_pred = model.predict(x)[0] > 0.5
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred.astype(np.int32)

    """ Saving the predicted mask """
    save_image_path = f"D:/MMST/s2(22)/Machine learning/webApp/static/masks/{imageName}"
    mask = save_results(y_pred, save_image_path)
    mask = np.asarray(mask).flatten()
    return mask
   



def getDisease(age,gender,area,imageName):
    knnModel = pickle.load(open("D:/MMST/s2(22)/Machine learning/webApp/static/model/Knn/model.pkl","rb"))
    row=[]
    row.append(int(age)), row.append(int(gender)), row.append(int(area))
    mask=np.asarray(ImageOps.grayscale(Image.open("D:/MMST/s2(22)/Machine learning/webApp/static/masks/"+imageName).resize((32,32))))
    print("==========> the row",row,len(mask))
    row = np.concatenate((row,mask.flatten()))
    result = knnModel.predict([row])
    return result