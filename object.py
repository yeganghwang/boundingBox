import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QListWidget, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLO 객체 인식 GUI (PyQt5)')
        self.setGeometry(100, 100, 900, 700)

        self.yolo_model = YOLO('yolov8n.pt')
        self.cv_img = None
        self.result_img = None
        self.last_results = None
        self.img_path = None

        # 위젯 생성
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet('background-color: white;')

        self.open_btn = QPushButton('파일 열기')
        self.detect_btn = QPushButton('객체 인식')
        self.save_btn = QPushButton('결과 저장')
        self.detect_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        self.result_list = QListWidget()
        self.result_list.setFixedHeight(150)

        # 레이아웃 구성
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.detect_btn)
        btn_layout.addWidget(self.save_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.img_label, stretch=1)
        main_layout.addWidget(QLabel('[객체 인식 결과]'))
        main_layout.addWidget(self.result_list)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 시그널 연결
        self.open_btn.clicked.connect(self.open_file)
        self.detect_btn.clicked.connect(self.detect)
        self.save_btn.clicked.connect(self.save_result)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '이미지 파일 열기', '', 'Image Files (*.jpg *.jpeg *.png)')
        if file_path:
            self.img_path = file_path
            self.cv_img = cv2.imread(file_path)
            if self.cv_img is not None:
                self.show_image(self.cv_img)
                self.detect_btn.setEnabled(True)
                self.save_btn.setEnabled(False)
                self.result_list.clear()

    def show_image(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pixmap)

    def detect(self):
        if self.cv_img is not None:
            results = self.yolo_model(self.cv_img)
            self.last_results = results[0]
            self.result_img = results[0].plot()
            self.show_image(self.result_img)
            self.save_btn.setEnabled(True)
            # 결과 리스트 표시
            self.result_list.clear()
            boxes = results[0].boxes
            names = results[0].names
            class_counts = {}
            for box in boxes:
                cls_id = int(box.cls[0])
                name = names[cls_id]
                conf = float(box.conf[0])
                class_counts.setdefault(name, []).append(conf)
                self.result_list.addItem(f"{name} (신뢰도: {conf:.2f})")
            self.result_list.addItem('----------------------')
            for name, confs in class_counts.items():
                self.result_list.addItem(f"{name}: {len(confs)}개")

    def save_result(self):
        if self.result_img is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, '결과 이미지 저장', '', 'JPEG Files (*.jpg);;PNG Files (*.png)')
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.result_img, cv2.COLOR_RGB2BGR))
                QMessageBox.information(self, '저장 완료', f'결과 이미지가 저장되었습니다:\n{file_path}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())