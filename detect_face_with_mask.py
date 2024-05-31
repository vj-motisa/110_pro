# import the opencv library
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("C:/Users/Admin/OneDrive/Desktop/Python/PRO-C110-Student-Boilerplate-main/keras_model.h5")



# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

    img = cv2.resize(frame,(224,224))
    i1 = np.array(img, dtype=np.float32)
    i2 = np.expand_dims(i1, axis=0)
    n_img = i2/255.0
    prediction = model.predict(n_img)
    predict_class = np.argmax(prediction, axis=1)
    print("Prediction: ", predict_class)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()