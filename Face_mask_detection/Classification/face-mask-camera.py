import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

model = keras.models.load_model(r'CNN.h5')

def check_mask(master_I):
    mask_status = {
        0: 'No',
        1: 'Yes'
    }

    orig_image = cv2.cvtColor(master_I, cv2.COLOR_RGB2GRAY)
    is_mask = 0

    try:
        # preprocessing
        image = cv2.resize(orig_image, (50, 50))
        image = image.reshape((*image.shape, 1))
        image = tf.convert_to_tensor(image)
        image = tf.image.grayscale_to_rgb(image).numpy() / 255.  # to scale image from 0 to 1
        final_image = image.reshape(1, *image.shape)
        prediction = model.predict(final_image)
        is_mask = tf.math.argmax(prediction, 1).numpy()[0]
        mask_probability = prediction[0][1] * 100  # Probability of having a mask (%)
    except Exception as E:
        print(E)
    else:
        master_I = cv2.putText(master_I,
                               f'Mask: {mask_status[is_mask]} - Probability: {mask_probability:.2f}%',
                               (10, 50),
                               fontFace=cv2.FONT_HERSHEY_DUPLEX,
                               fontScale=1,
                               color=(0, 0, 255), thickness=2)

    return master_I

cap = cv2.VideoCapture(0)  # live
# cap = cv2.VideoCapture('demo.mp4')
if not cap.isOpened():
    print("There is an error in loading video file.")

while True:
    ret, frame = cap.read()
    if ret:  # ret is TRUE only till video can be read
        time.sleep(1 / cap.get(cv2.CAP_PROP_FPS))  # to get the fps of video
        frame = check_mask(frame)
        cv2.imshow("title", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
