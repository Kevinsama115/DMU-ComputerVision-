
import cv2 as cv
import numpy as np
import tensorflow as tf
import pickle
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton,
    QFileDialog, QLabel, QMessageBox
)

# Load trained model
cnn = tf.keras.models.load_model('cnn_for_stanford_dogs.h5')

# Load dog breed names
with open('dog_species_names.txt', 'rb') as f:
    dog_species = pickle.load(f)


class DogSpeciesRecognition(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle('Dog Recognition')
        self.setGeometry(200, 200, 700, 300)

        # Buttons
        fileButton = QPushButton('강아지 사진 열기', self)
        recognitionButton = QPushButton('품종 인식', self)
        quitButton = QPushButton('나가기', self)

        # Button positions
        fileButton.setGeometry(10, 10, 180, 30)
        recognitionButton.setGeometry(200, 10, 100, 30)
        quitButton.setGeometry(550, 10, 100, 30)

        # Image label
        self.imgLabel = QLabel(self)
        self.imgLabel.setGeometry(10, 60, 300, 200)

        # Button events
        fileButton.clicked.connect(self.pictureOpenFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.img = None

    # Open dog picture
    def pictureOpenFunction(self):

        fname = QFileDialog.getOpenFileName(
            self,
            '강아지 사진 열기',
            './'
        )

        if fname[0]:
            self.img = cv.imread(fname[0])

            if self.img is None:
                QMessageBox.warning(
                    self,
                    'Error',
                    '이미지를 열 수 없습니다.'
                )
                return

            self.displayImage()

    # Display image
    def displayImage(self):

        rgb = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        bytesPerLine = ch * w

        from PyQt5.QtGui import QImage, QPixmap

        qimg = QImage(
            rgb.data,
            w,
            h,
            bytesPerLine,
            QImage.Format_RGB888
        )

        self.imgLabel.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                300, 200
            )
        )

    # Recognize dog breed
    def recognitionFunction(self):

        if self.img is None:
            QMessageBox.warning(
                self,
                'Error',
                '먼저 강아지 사진을 선택하세요.'
            )
            return

        # Resize image
        x = cv.resize(self.img, (224, 224))

        # Convert BGR to RGB
        x = cv.cvtColor(x, cv.COLOR_BGR2RGB)

        # Normalize image
        x = x.astype('float32') / 255.0

        # Reshape image
        x = np.reshape(x, (1, 224, 224, 3))

        # Predict
        res = cnn.predict(x, verbose=0)[0]

        # Get top 5 predictions
        top5 = np.argsort(res)[-5:][::-1]

        result = ''

        for i in top5:
            result += (
                dog_species[i] + ': '
                + str(round(float(res[i]), 4))
                + '\n'
            )

        QMessageBox.information(
            self,
            'Dog Recognition Result',
            result
        )

    # Quit program
    def quitFunction(self):
        self.close()


# Run application
app = QApplication(sys.argv)
win = DogSpeciesRecognition()
win.show()
sys.exit(app.exec_())
