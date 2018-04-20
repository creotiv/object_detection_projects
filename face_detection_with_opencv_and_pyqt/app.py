import sys
from os import path

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui


class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)


class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, cascade_filepath_front, cascade_filepath_prof, parent=None):
        super().__init__(parent)
        self.classifier_1 = cv2.CascadeClassifier(cascade_filepath_front)
        self.classifier_2 = cv2.CascadeClassifier(cascade_filepath_prof)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (30, 30)

    def detect_faces(self, image: np.ndarray):
        # haarclassifiers work better in black and white
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

    def image_data_slot(self, image_data):
        faces = self.detect_faces(image_data)
        for (x, y, w, h) in faces:
            cv2.rectangle(image_data,
                          (x, y),
                          (x+w, y+h),
                          self._red,
                          self._width)

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
        self.face_detection_widget = FaceDetectionWidget(cascade_filepath_front, cascade_filepath_prof)

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
