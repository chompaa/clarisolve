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
)

import models
from colorize import colorize
from super_resolve import super_resolve


class ScaleSelector(QGroupBox):
    scale = None

    def __init__(self, scale_levels, scale_default):
        super(QGroupBox, self).__init__()

        self.scale = scale_default
        self.setTitle("Scale")

        layout = QHBoxLayout()

        for scale in scale_levels:
            toggle = QRadioButton(str(scale))

            if scale == scale_default:
                toggle.setChecked(True)

            toggle.toggled.connect(self.on_selected)

            layout.addWidget(toggle)

        layout.addStretch(1)

        self.setLayout(layout)

    def get_scale(self):
        return int(self.scale)

    def on_selected(self):
        scale_button = self.sender()

        if not scale_button or not (type(scale_button) is QRadioButton):
            return

        if scale_button.isChecked():
            self.scale = int(scale_button.text())


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


class FileSelector(QGroupBox):
    def __init__(self, title, default_path, file_type):
        super(QGroupBox, self).__init__()

        self.setTitle(title)
        self.file_type = file_type

        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.path_input = QLineEdit(default_path)

        select_file = QPushButton("Choose File")
        select_file.clicked.connect(self.select_file)

        layout.addWidget(self.path_input)
        layout.addWidget(select_file)

        self.setLayout(layout)

    def select_file(self):
        file_path = QFileDialog.getOpenFileName(
            self, "Select File", "", self.file_type
        )[0]

        if file_path:
            self.path_input.setText(file_path)

    def get_file(self):
        return self.path_input.text()


class StateSelector(QGroupBox):
    def __init__(self, title, initial_state):
        super(QGroupBox, self).__init__()

        self.setTitle(title)
        self.state = initial_state

        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        yes_button = QCheckBox("Yes")
        yes_button.setChecked(initial_state)
        yes_button.toggled.connect(self.on_selected)

        layout.addWidget(yes_button)

        self.setLayout(layout)

    def on_selected(self):
        button = self.sender()

        if not button or not (type(button) is QCheckBox):
            return

        if button.isChecked():
            self.state = True
        else:
            self.state = False

    def get_state(self):
        return self.state


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

        self.setWindowTitle("Clary-solve")

        widget = QWidget()

        main_layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        scale_colorize_layout = QHBoxLayout()
        top_left_layout = QVBoxLayout()
        top_right_layout = QVBoxLayout()

        bottom_layout = QHBoxLayout()

        self.colorize_selector = StateSelector("Colorize", True)
        self.scale_selector = ScaleSelector(["1", "2", "3", "4"], "2")
        self.sisr_weights_selector = FileSelector(
            "Super-resolution weights",
            # abspath("./weights/x2/best.pth"),
            "",
            "PyTorch State Dictionary Files *.pth",
        )
        self.sic_weights_selector = FileSelector(
            "Colorization weights",
            # abspath("./weights/ic.pth"),
            "",
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

        scale_colorize_layout.addWidget(self.colorize_selector)
        scale_colorize_layout.addWidget(self.scale_selector)
        scale_colorize_layout.addStretch(1)

        top_left_layout.addLayout(scale_colorize_layout)
        top_left_layout.addWidget(self.sisr_weights_selector)
        top_left_layout.addWidget(self.sic_weights_selector)
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
        scale = self.scale_selector.get_scale()

        open_on_complete = self.open_on_complete.isChecked()

        colorize_image = self.colorize_selector.get_state()

        if colorize_image:
            colorize(
                models.IC_MODELS["icres"](), sic_weights, output_folder, image_path
            )
            image_name, image_ext = os.path.splitext(os.path.basename(image_path))
            image_path = os.path.join(
                output_folder, f"{image_name}_colorized{image_ext}"
            )

        if scale != 1:
            super_resolve(
                models.SR_MODELS["srcnn"](),
                sisr_weights,
                output_folder,
                image_path,
                scale,
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
