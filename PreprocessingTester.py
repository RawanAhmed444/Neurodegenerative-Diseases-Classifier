import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox, QFileDialog
from PyQt5.QtGui import QPixmap, QImage

class ImagePreprocessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_index = 0
        self.image_paths = self.load_images("Data/parkinson")
        self.initUI()
        self.load_image()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Image Labels
        self.original_label = QLabel("Original Image")
        self.processed_label = QLabel("Processed Image")
        
        layout.addWidget(self.original_label)
        layout.addWidget(self.processed_label)
        
        # Checkboxes for preprocessing
        self.clahe_cb = QCheckBox("Apply CLAHE")
        self.hist_eq_cb = QCheckBox("Histogram Equalization")
        self.denoise_cb = QCheckBox("Noise Reduction")
        self.sharpen_cb = QCheckBox("Sharpening")
        
        layout.addWidget(self.clahe_cb)
        layout.addWidget(self.hist_eq_cb)
        layout.addWidget(self.denoise_cb)
        layout.addWidget(self.sharpen_cb)
        
        # Buttons
        self.process_button = QPushButton("Apply Preprocessing")
        self.process_button.clicked.connect(self.apply_preprocessing)
        
        self.next_button = QPushButton("Next Image")
        self.next_button.clicked.connect(self.next_image)
        
        layout.addWidget(self.process_button)
        layout.addWidget(self.next_button)
        
        self.setLayout(layout)
        self.setWindowTitle("Image Preprocessing App")

    def load_images(self, folder):
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg'))]
    
    def load_image(self):
        if self.image_index < len(self.image_paths):
            img_path = self.image_paths[self.image_index]
            self.original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.original_label, self.original_img)
    
    def apply_preprocessing(self):
        if self.original_img is None:
            return
        processed_img = self.original_img.copy()

        if self.clahe_cb.isChecked():
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            processed_img = clahe.apply(processed_img)

        if self.hist_eq_cb.isChecked():
            processed_img = cv2.equalizeHist(processed_img)

        if self.denoise_cb.isChecked():
            processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)
        
        if self.sharpen_cb.isChecked():
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed_img = cv2.filter2D(processed_img, -1, kernel)

        self.display_image(self.processed_label, processed_img)
    
    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.image_paths)
        self.load_image()
        self.processed_label.clear()
    
    def display_image(self, label, img):
        height, width = img.shape
        bytes_per_line = width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImagePreprocessorApp()
    window.show()
    sys.exit(app.exec_())
