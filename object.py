import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QListWidget, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YoloV8 객체인식')
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet('background-color: #f6f6fa;')

        self.yolo_model = YOLO('yolov8n.pt')
        self.cv_img = None
        self.result_img = None
        self.last_results = None
        self.img_path = None
        self.boxes_info = []  # (box, name, conf) 저장

        # 타이틀 라벨
        self.title_label = QLabel('YOLO 객체인식')
        self.title_label.setFont(QFont('Arial', 22, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet('color: #22223b; margin: 20px 0 10px 0;')

        # 위젯 생성
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet('background-color: #fff; border-radius: 12px; border: 1.5px solid #e0e0e0; margin: 10px;')

        self.open_btn = QPushButton('파일 열기')
        self.detect_btn = QPushButton('객체 인식')
        self.save_btn = QPushButton('결과 저장')
        for btn in [self.open_btn, self.detect_btn, self.save_btn]:
            btn.setFixedHeight(40)
            btn.setStyleSheet('''
                QPushButton {
                    background-color: #4f8cff;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 16px;
                    padding: 0 24px;
                }
                QPushButton:disabled {
                    background-color: #bfc9d1;
                    color: #f6f6fa;
                }
                QPushButton:hover:!disabled {
                    background-color: #2563eb;
                }
            ''')
        self.detect_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

        self.result_list = QListWidget()
        self.result_list.setFixedHeight(180)
        self.result_list.setStyleSheet('''
            QListWidget {
                background: #fff;
                border-radius: 10px;
                border: 1.5px solid #e0e0e0;
                font-size: 15px;
                padding: 8px;
            }
            QListWidget::item:selected {
                background: #e0e7ff;
                color: #22223b;
            }
        ''')

        # 구분선
        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.line.setStyleSheet('color: #bfc9d1; margin: 10px 0;')

        # 레이아웃 구성
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.open_btn)
        btn_layout.addWidget(self.detect_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch(1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.img_label, stretch=1)
        main_layout.addWidget(self.line)
        main_layout.addWidget(QLabel('[객체 인식 결과]'))
        main_layout.addWidget(self.result_list)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(12)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 시그널 연결
        self.open_btn.clicked.connect(self.open_file)
        self.detect_btn.clicked.connect(self.detect)
        self.save_btn.clicked.connect(self.save_result)
        self.result_list.itemClicked.connect(self.highlight_box)

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
                self.boxes_info = []

    def show_image(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # 화면에 표시할 때만 최대 크기로 resize (원본은 그대로 보존)
        max_width, max_height = 450, 300
        if w > max_width or h > max_height:
            pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pixmap)

    def detect(self):
        if self.cv_img is not None:
            results = self.yolo_model(self.cv_img)
            self.last_results = results[0]
            self.result_img = results[0].plot()
            self.show_image(self.result_img)  # 화면에는 resize된 결과만 표시
            self.save_btn.setEnabled(True)
            # 결과 리스트 표시
            self.result_list.clear()
            boxes = results[0].boxes
            names = results[0].names
            class_counts = {}
            self.boxes_info = []
            for box in boxes:
                cls_id = int(box.cls[0])
                name = names[cls_id]
                conf = float(box.conf[0])
                class_counts.setdefault(name, []).append(conf)
                self.result_list.addItem(f"{name} (신뢰도: {conf:.2f})")
                self.boxes_info.append((box, name, conf))
            self.result_list.addItem('----------------------')
            for name, confs in class_counts.items():
                self.result_list.addItem(f"{name}: {len(confs)}개")

    def highlight_box(self, item):
        idx = self.result_list.row(item)
        # 클래스별 개수 요약 구분선 이후는 무시
        if idx >= len(self.boxes_info):
            return
        box, name, conf = self.boxes_info[idx]
        # 원본 이미지 복사
        img = self.cv_img.copy()
        # 바운딩 박스 좌표 추출
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        # 바운딩 박스 그리기 (굵은 빨간색)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
        # 라벨 텍스트 그리기
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.rectangle(img, (x1, y1 - th - 12), (x1 + tw, y1), (0, 0, 255), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        self.show_image(img)  # 화면에는 resize된 결과만 표시

    def save_result(self):
        if self.result_img is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, '결과 이미지 저장', '', 'JPEG Files (*.jpg);;PNG Files (*.png)')
            if file_path:
                # result_img는 화면 표시용이므로, 원본 해상도로 다시 추론해서 저장
                results = self.yolo_model(self.cv_img)
                orig_result_img = results[0].plot()
                cv2.imwrite(file_path, cv2.cvtColor(orig_result_img, cv2.COLOR_RGB2BGR))
                QMessageBox.information(self, '저장 완료', f'결과 이미지가 저장되었습니다:\n{file_path}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())