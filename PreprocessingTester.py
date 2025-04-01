import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QCheckBox, QComboBox, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage

class ImagePreprocessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_index = 0
        self.classes = ["normal", "alzheimer", "parkinson"]  
        self.base_folder = "Data"  
        self.image_paths = []  
        self.processed_img = None
        self.initUI()
        self.load_images_for_class(self.classes[0])  
        self.load_image()

    def initUI(self):
        main_layout = QVBoxLayout()

        # ComboBox for selecting class
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.classes) 
        self.class_combo.currentIndexChanged.connect(self.on_class_changed) 
        main_layout.addWidget(QLabel("Select Class:"))
        main_layout.addWidget(self.class_combo)

        # Image Labels with Titles in Horizontal Layout
        images_layout = QHBoxLayout()
        
        # Original Image section
        original_section = QVBoxLayout()
        original_title = QLabel("Unpreprocessed Image")
        original_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.original_label = QLabel()
        self.original_label.setMinimumSize(300, 300)
        original_section.addWidget(original_title)
        original_section.addWidget(self.original_label)
        
        # Processed Image section
        processed_section = QVBoxLayout()
        processed_title = QLabel("Preprocessed Image")
        processed_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.processed_label = QLabel()
        self.processed_label.setMinimumSize(300, 300)
        processed_section.addWidget(processed_title)
        processed_section.addWidget(self.processed_label)
        
        images_layout.addLayout(original_section)
        images_layout.addLayout(processed_section)
        main_layout.addLayout(images_layout)
        
        # Checkboxes for preprocessing
        self.clahe_cb = QCheckBox("Apply CLAHE")
        self.hist_eq_cb = QCheckBox("Histogram Equalization")
        self.denoise_cb = QCheckBox("Noise Reduction")
        self.sharpen_cb = QCheckBox("Sharpening")
        self.recommended_pipeline_cb = QCheckBox("Use Recommended Pipeline (Noise → CLAHE → Subtle Sharpening)")
        
        main_layout.addWidget(self.clahe_cb)
        main_layout.addWidget(self.hist_eq_cb)
        main_layout.addWidget(self.denoise_cb)
        main_layout.addWidget(self.sharpen_cb)
        main_layout.addWidget(self.recommended_pipeline_cb)

        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.process_button = QPushButton("Apply Preprocessing")
        self.process_button.clicked.connect(self.apply_preprocessing)
        
        self.next_button = QPushButton("Next Image")
        self.next_button.clicked.connect(self.next_image)
        
        self.download_button = QPushButton("Download Comparison")
        self.download_button.clicked.connect(self.download_comparison)
        self.download_button.setEnabled(False)  # Disable until processing is done
        
        buttons_layout.addWidget(self.process_button)
        buttons_layout.addWidget(self.next_button)
        buttons_layout.addWidget(self.download_button)
        
        main_layout.addLayout(buttons_layout)
        
        self.setLayout(main_layout)
        self.setWindowTitle("Image Preprocessing App")
        self.resize(700, 600)

    def load_images_for_class(self, class_name):
        """Load images for the selected class."""
        folder = os.path.join(self.base_folder, class_name)
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist!")
            self.image_paths = []
            return
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.image_index = 0  
        if not self.image_paths:
            print(f"No images found in {folder}!")
            self.original_label.clear()
            self.processed_label.clear()

    def on_class_changed(self):
        """Handle class selection change from ComboBox."""
        selected_class = self.class_combo.currentText()
        self.load_images_for_class(selected_class)
        self.load_image()
        self.processed_img = None
        self.processed_label.clear()
        self.download_button.setEnabled(False)

    def load_image(self):
        """Load the current image based on the image index."""
        if self.image_paths and self.image_index < len(self.image_paths):
            img_path = self.image_paths[self.image_index]
            self.original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if self.original_img is not None:
                self.display_image(self.original_label, self.original_img)
                self.processed_img = None
                self.processed_label.clear()
                self.download_button.setEnabled(False)
            else:
                print(f"Failed to load image: {img_path}")
                self.original_label.clear()
        else:
            self.original_label.clear()

    def apply_preprocessing(self):
        """Apply selected preprocessing techniques to the image."""
        if self.original_img is None:
            return
        self.processed_img = self.original_img.copy()

        # Check if recommended pipeline is selected
        if self.recommended_pipeline_cb.isChecked():
            # 1. Noise Reduction with Gaussian blur
            self.processed_img = cv2.GaussianBlur(self.processed_img, (5,5), 0)
            
            # 2. Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            self.processed_img = clahe.apply(self.processed_img)
            
            # 3. Subtle Sharpening with gentler kernel
            gentle_kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
            self.processed_img = cv2.filter2D(self.processed_img, -1, gentle_kernel)
        else:
            # Original individual preprocessing options
            if self.clahe_cb.isChecked():
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                self.processed_img = clahe.apply(self.processed_img)

            if self.hist_eq_cb.isChecked():
                self.processed_img = cv2.equalizeHist(self.processed_img)

            if self.denoise_cb.isChecked():
                self.processed_img = cv2.GaussianBlur(self.processed_img, (5,5), 0)
            
            if self.sharpen_cb.isChecked():
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                self.processed_img = cv2.filter2D(self.processed_img, -1, kernel)

        self.display_image(self.processed_label, self.processed_img)
        self.download_button.setEnabled(True)
    
    def next_image(self):
        """Load the next image in the list."""
        if self.image_paths:
            self.image_index = (self.image_index + 1) % len(self.image_paths)
            self.load_image()
            self.processed_img = None
            self.processed_label.clear()
            self.download_button.setEnabled(False)
    
    def display_image(self, label, img):
        """Display an image in a QLabel."""
        height, width = img.shape
        bytes_per_line = width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def download_comparison(self):
        """Create and save a high-quality, professional side-by-side comparison of original and processed images."""
        if self.original_img is None or self.processed_img is None:
            return
            
        # Get current image filename for default save name
        if self.image_paths and self.image_index < len(self.image_paths):
            base_filename = os.path.basename(self.image_paths[self.image_index])
            default_save_name = f"comparison_{base_filename}"
        else:
            default_save_name = "image_comparison.png"
            
        # Open file dialog for save location
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Comparison Image", default_save_name, "Images (*.png *.jpg)"
        )
        
        if not file_path:
            return  # User canceled the dialog
            
        # Create a high-resolution comparison image
        height, width = self.original_img.shape
        
        # Ensure both images are the same size (in case of any processing that changes dimensions)
        processed_resized = cv2.resize(self.processed_img, (width, height))
        
        # Set parameters for professional formatting
        margin = 50  # Increased margin for better spacing
        title_margin = 30  # Space between title and image
        bottom_margin = 40  # Space for the methods text at bottom
        title_height = 50  # Height allocation for title text
        
        # Calculate needed canvas size
        total_width = width * 2 + margin * 3
        total_height = height + title_height + title_margin + bottom_margin + margin
        
        # Create a blank white image with proper dimensions
        comparison = np.ones((total_height, total_width), dtype=np.uint8) * 255
        
        # Font settings - using a cleaner font with better spacing
        font = cv2.FONT_HERSHEY_DUPLEX
        title_font_scale = 1.0
        methods_font_scale = 0.7
        title_thickness = 2
        methods_thickness = 1
        text_color = 0  # Black text
        
        # Calculate text positions for proper centering
        title1 = "Unpreprocessed Image"
        title2 = "Preprocessed Image"
        
        # Center text positions for titles
        title1_size = cv2.getTextSize(title1, font, title_font_scale, title_thickness)[0]
        title2_size = cv2.getTextSize(title2, font, title_font_scale, title_thickness)[0]
        
        title1_x = margin + (width - title1_size[0]) // 2
        title2_x = margin * 2 + width + (width - title2_size[0]) // 2
        title_y = margin + title_height // 2
        
        # Draw titles with proper centering
        cv2.putText(comparison, title1, (title1_x, title_y), 
                    font, title_font_scale, text_color, title_thickness)
        cv2.putText(comparison, title2, (title2_x, title_y), 
                    font, title_font_scale, text_color, title_thickness)
        
        # Calculate image positions
        img_y_start = margin + title_height + title_margin
        
        # Place original image
        comparison[img_y_start:img_y_start + height, 
                margin:margin + width] = self.original_img
        
        # Place processed image
        comparison[img_y_start:img_y_start + height, 
                margin * 2 + width:margin * 2 + width * 2] = processed_resized
        
        # Add thin borders around images for cleaner separation
        cv2.rectangle(comparison, 
                    (margin, img_y_start), 
                    (margin + width, img_y_start + height), 
                    text_color, 1)
        cv2.rectangle(comparison, 
                    (margin * 2 + width, img_y_start), 
                    (margin * 2 + width * 2, img_y_start + height), 
                    text_color, 1)
        
        # Add preprocessing methods used at the bottom
        methods_text = "Methods: "
        if self.clahe_cb.isChecked():
            methods_text += "CLAHE"
        if self.hist_eq_cb.isChecked():
            methods_text += ", Histogram Equalization" if methods_text != "Methods: " else "Histogram Equalization"
        if self.denoise_cb.isChecked():
            methods_text += ", Noise Reduction" if methods_text != "Methods: " else "Noise Reduction"
        if self.sharpen_cb.isChecked():
            methods_text += ", Sharpening" if methods_text != "Methods: " else "Sharpening"
        
        if methods_text == "Methods: ":
            methods_text += "None"
        
        # Center methods text
        methods_size = cv2.getTextSize(methods_text, font, methods_font_scale, methods_thickness)[0]
        methods_x = (total_width - methods_size[0]) // 2
        methods_y = img_y_start + height + bottom_margin // 2
        
        cv2.putText(comparison, methods_text, (methods_x, methods_y), 
                font, methods_font_scale, text_color, methods_thickness)
        
        # Save the comparison image
        cv2.imwrite(file_path, comparison)
        print(f"Comparison saved to {file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImagePreprocessorApp()
    window.show()
    sys.exit(app.exec_())