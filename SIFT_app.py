#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import Image_tracker as img

import cv2
import sys

class My_App(QtWidgets.QMainWindow):
	
    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
             self.template_path = dlg.selectedFiles()[0]
             self._image_matcher = img.ImageComparer(cv2.imread(self.template_path))

        pixMap = QtGui.QPixmap(self.template_path)
        
        self.template_label.setPixmap(pixMap)

        print('loaded template image file: ' + self.template_path)
		
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./Sift_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 25
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)
        self.browse_button.clicked.connect(self.SLOT_browse_button)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 600)
        self._camera_device.set(4, 240)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)
    
    def convert_cv_to_pixmap(self, cv_img):
         cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
         height, width, channel = cv_img.shape
         bytesPerLine = channel * width
         q_img = QtGui.QImage(cv_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
         return QtGui.QPixmap.fromImage(q_img)
    
    def SLOT_query_camera(self):
         ret, frame = self._camera_device.read()

         keyPointed = self._image_matcher.matchFrame(frame)

         pixmap = self.convert_cv_to_pixmap(keyPointed)
         self.live_image_label.setPixmap(pixmap)


    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable Camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("Enable Camera")



	
    
		

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	sys.exit(app.exec_())

