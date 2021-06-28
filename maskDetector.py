import cvlib as cv
import cv2
import numpy as np
import tensorflow as tf

import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# GPIO pin number setting
red_led_pin = 16
blue_led_pin = 20
GPIO.setup(red_led_pin, GPIO.OUT)
GPIO.setup(blue_led_pin, GPIO.OUT)

buzzer_pin = 18
GPIO.setup(buzzer_pin, GPIO.OUT)  # buzzer
p = GPIO.PWM(buzzer_pin, 80)

model = tf.keras.models.load_model('model.h5')
model.summary() # print model structure

camera = cv2.VideoCapture(0)

while camera.isOpened():

    frame = camera.read()
    face, confidence = cv.detect_face(frame)

    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[
            0] and 0 <= endY <= frame.shape[0]:

            detected_face = frame[startY:endY, startX:endX]

            resized_face = cv2.resize(detected_face, (224, 224), interpolation=cv2.INTER_AREA)

            x = tf.keras.preprocessing.image.img_to_array(resized_face)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

            prediction = model.predict(x)

            if prediction < 0.5:  # without mask
                p.start(10)
                p.ChangeFrequency(80)
                GPIO.output(red_led_pin, True)
                GPIO.output(blue_led_pin, False)

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "No Mask ({:.2f}%)".format((1 - prediction[0][0]) * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:  # with mask
                p.stop()
                GPIO.output(red_led_pin, False)
                GPIO.output(blue_led_pin, True)

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Mask ({:.2f}%)".format(prediction[0][0] * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("mask detector", frame)

    # stop detecting
    if cv2.waitKey(1) & 0xFF == ord('q'):
        p.stop()
        GPIO.cleanup()  # initialize gpio setting

        break

# release resources
camera.release()
cv2.destroyAllWindows()

