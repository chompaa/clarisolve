import os
import sys
from os import startfile
from os.path import abspath
from platform import system
from subprocess import Popen

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

import models
from colorize import colorize
from super_resolve import super_resolve


class FolderSelector(QGroupBox):
    def __init__(self, title, default_path):
        super(QGroupBox, self).__init__()

        self.setTitle(title)

        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.path_input = QLineEdit(default_path + "/")

        select_folder = QPushButton("Choose Folder")
        select_folder.clicked.connect(self.select_folder)

        layout.addWidget(self.path_input)
        layout.addWidget(select_folder)

        self.setLayout(layout)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", "") + "/"

        if folder_path:
            self.path_input.setText(folder_path)

    def get_folder(self):
        return self.path_input.text()


class FileSelector(QWidget):
    def __init__(self, placeholder_text, file_type):
        super(QWidget, self).__init__()

        self.file_type = file_type

        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText(placeholder_text)

        select_file = QPushButton("Choose File")
        select_file.clicked.connect(self.select_file)

        layout.addWidget(self.path_input)
        layout.addWidget(select_file)
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)

    def select_file(self):
        file_path = QFileDialog.getOpenFileName(
            self, "Select File", "", self.file_type
        )[0]

        if file_path:
            self.path_input.setText(file_path)

    def get_file(self):
        return self.path_input.text()


class ImageSelector(QGroupBox):
    image_path = abspath("./butterfly.png")

    def __init__(self, image_width, image_height):
        super(QWidget, self).__init__()

        self.image_width = image_width
        self.image_height = image_height

        self.setTitle("Select image")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.display_image = QLabel()
        self.display_image.setScaledContents(True)

        self.set_image(self.image_path)

        select_image = QPushButton("Choose file")
        select_image.clicked.connect(self.select_image)
        select_image.setFixedWidth(self.image_width)

        layout.addWidget(self.display_image, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(select_image, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

        self.setLayout(layout)

    def set_image(self, path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(
            QSize(self.image_width, self.image_height),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.display_image.setFixedWidth(pixmap.width())
        self.display_image.setFixedHeight(pixmap.height())
        self.display_image.setPixmap(pixmap)

    def select_image(self):
        self.image_path = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )[0]

        if self.image_path:
            self.set_image(self.image_path)

    def get_image_path(self):
        return self.image_path


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Clarisolve")

        widget = QWidget()

        main_layout = QVBoxLayout()

        top_layout = QHBoxLayout()

        super_resolve_group = QGroupBox()
        super_resolve_group.setTitle("Super Resolution")
        super_resolve_layout = QVBoxLayout()
        super_resolve_group.setLayout(super_resolve_layout)

        colorize_group = QGroupBox()
        colorize_group.setTitle("Colorization")
        colorize_layout = QVBoxLayout()
        colorize_group.setLayout(colorize_layout)

        top_left_layout = QVBoxLayout()
        top_right_layout = QVBoxLayout()

        bottom_layout = QHBoxLayout()

        self.super_resolve_model = QComboBox()
        self.super_resolve_model.addItems(models.SR_MODELS.keys())
        self.super_resolve_model.setPlaceholderText("Select model")
        self.super_resolve_model.setCurrentIndex(-1)

        self.colorize_model = QComboBox()
        self.colorize_model.addItems(models.IC_MODELS.keys())
        self.colorize_model.setPlaceholderText("Select model")
        self.colorize_model.setCurrentIndex(-1)

        self.sisr_weights_selector = FileSelector(
            "Path to weights..",
            "PyTorch State Dictionary Files *.pth",
        )
        self.sic_weights_selector = FileSelector(
            "Path to weights..",
            "PyTorch State Dictionary Files *.pth",
        )
        self.output_selector = FolderSelector(
            "Output folder", abspath("./output").replace("\\", "/")
        )
        self.input_selector = ImageSelector(190, 190)

        self.status = QLabel("Ready")
        self.open_on_complete = QCheckBox("Open on complete")
        self.open_on_complete.setChecked(True)

        bottom_layout.addWidget(self.status, alignment=Qt.AlignmentFlag.AlignLeft)
        bottom_layout.addWidget(
            self.open_on_complete, alignment=Qt.AlignmentFlag.AlignRight
        )

        enhance_button = QPushButton("Enhance")
        enhance_button.clicked.connect(self.enhance)

        super_resolve_layout.addWidget(self.super_resolve_model)
        super_resolve_layout.addWidget(self.sisr_weights_selector)
        super_resolve_layout.addStretch(1)

        colorize_layout.addWidget(self.colorize_model)
        colorize_layout.addWidget(self.sic_weights_selector)
        colorize_layout.addStretch(1)

        top_left_layout.addWidget(super_resolve_group)
        top_left_layout.addWidget(colorize_group)
        top_left_layout.addWidget(self.output_selector)

        top_right_layout.addWidget(self.input_selector)
        top_right_layout.addStretch(1)

        top_layout.addLayout(top_left_layout)
        top_layout.addLayout(top_right_layout)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)
        main_layout.addWidget(enhance_button, alignment=Qt.AlignmentFlag.AlignBottom)

        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        self.show()

    def enhance(self):
        sisr_weights = self.sisr_weights_selector.get_file()
        sic_weights = self.sic_weights_selector.get_file()
        output_folder = self.output_selector.get_folder()
        image_path = self.input_selector.get_image_path()

        super_resolve_model = self.super_resolve_model.currentText()
        colorize_model = self.colorize_model.currentText()

        open_on_complete = self.open_on_complete.isChecked()

        if colorize_model:
            if not os.path.exists(sic_weights):
                self.set_status("Colorization weights not found")
                return

            colorize(
                models.IC_MODELS[colorize_model](),
                sic_weights,
                output_folder,
                image_path,
            )
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            image_path = os.path.join(
                output_folder, f"{image_name}_colorized{image_ext}"
            )

        if super_resolve_model:
            if not os.path.exists(sisr_weights):
                self.set_status("Super resolution weights not found")
                return

            super_resolve(
                models.SR_MODELS[super_resolve_model](),
                sisr_weights,
                output_folder,
                image_path,
                2,
                True,
            )

        if not open_on_complete:
            return

        if system() == "Windows":
            startfile(output_folder)
        elif system() == "Darwin":
            Popen(["open", output_folder])
        else:
            Popen(["xdg-open", output_folder])

    def set_status(self, status):
        self.status.setText(status)


def main():
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
