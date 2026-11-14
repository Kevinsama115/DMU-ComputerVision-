
import cv2 as cv
import numpy as np

from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions
)

# Load pretrained ResNet50 model
model = ResNet50(weights='imagenet')

# Read image
img = cv.imread('rabbit.jpg')

# Resize image to 224 x 224
x = np.reshape(cv.resize(img, (224, 224)), (1, 224, 224, 3))

# Preprocess image
x = preprocess_input(x)

# Predict
preds = model.predict(x)

# Get top 5 predictions
top5 = decode_predictions(preds, top=5)[0]

print('Prediction results:', top5)

# Display prediction results on image
for i in range(5):
    cv.putText(
        img,
        top5[i][1] + ': ' + str(top5[i][2]),
        (10, 20 + i * 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

cv.imshow('Recognition result', img)

cv.waitKey(0)
cv.destroyAllWindows()
