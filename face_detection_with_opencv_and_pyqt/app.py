import sys
from os import path

import cv2
import numpy as np
from classifier import FaceClass

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import time


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(tuple)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return
        
        t = time.time()
        read, data = self.camera.read()
        if read:
            self.image_data.emit((data, t))


class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, cascade_filepath_front, cascade_filepath_prof, fc, parent=None):
        super().__init__(parent)
        self.fc = fc
        self.classifier_1 = cv2.CascadeClassifier(cascade_filepath_front)
        self.classifier_2 = cv2.CascadeClassifier(cascade_filepath_prof)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    def detect_faces(self, image: np.ndarray):
        # haarclassifiers work better in black and white
        #image = cv2.resize(image,None, fx=1/2, fy=1/2, interpolation = cv2.INTER_AREA)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)

        faces_front = self.classifier_1.detectMultiScale(gray_image,
                                                 scaleFactor=1.3,
                                                 minNeighbors=4,
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 minSize=self._min_size)
        faces_prof = self.classifier_2.detectMultiScale(gray_image,
                                                 scaleFactor=1.3,
                                                 minNeighbors=4,
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 minSize=self._min_size)
        
        return faces_prof if isinstance(faces_prof,np.ndarray) else faces_front

    def image_data_slot(self, data):
        image_data, t = data
        faces = self.detect_faces(image_data)
        fc_gen, fc_age = self.fc.classify(image_data, faces)
        fps = 1/(time.time() - t)
        for i, (x, y, w, h) in enumerate(faces):
            #x, y, w, h = int(x*2), int(y*2), int(w*2), int(h*2)
            text = "Sex:%s Age:%s" % ('Male' if fc_gen[i] < 0.5 else 'Female', round(fc_age[i],1))
            cv2.rectangle(image_data,
                          (x, y),
                          (x+w, y+h),
                          self._red,
                          self._width)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_data,text,(x, y-5), font, 0.6,(255,255,255),1,cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_data,"FPS: {0:.2f}".format(fps),(5,20), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        self.image = self.get_qimage(image_data)
        # rescale window to image size
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)
                       
        # cv return image in BGR format
        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, cascade_filepath_front, cascade_filepath_prof, parent=None):
        super().__init__(parent)
        self.fc = FaceClass()
        self.face_detection_widget = FaceDetectionWidget(cascade_filepath_front, cascade_filepath_prof, self.fc)

        # TODO: set video port
        self.record_video = RecordVideo()

        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.face_detection_widget)
        self.run_button = QtWidgets.QPushButton('Start')
        layout.addWidget(self.run_button)

        self.run_button.clicked.connect(self.record_video.start_recording)
        self.setLayout(layout)


def main(cascade_filepath_front, cascade_filepath_prof):
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(cascade_filepath_front, cascade_filepath_prof)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cascade_filepath_front = path.join(script_dir, 'haarcascade_frontalface_default.xml')
    cascade_filepath_prof = path.join(script_dir, 'haarcascade_profileface.xml')

    cascade_filepath_front = path.abspath(cascade_filepath_front)
    cascade_filepath_prof = path.abspath(cascade_filepath_prof)
    main(cascade_filepath_front, cascade_filepath_prof)
