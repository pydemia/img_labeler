#!/usr/bin/env python3

import os
from pathlib import PurePath, Path
from glob import glob

import paramiko
import cv2
import numpy as np
import pandas as pd

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QSize, QModelIndex
from PyQt5.QtGui import (
    QImage, QPixmap, QPalette, QPainter, QFont, QKeySequence
)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import (
    QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction,
    qApp, QFileDialog,
    QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QComboBox, QRadioButton, QButtonGroup,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QLineEdit,
)

ssh = paramiko.SSHClient()

class QImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.printer = QPrinter()
        self.scaleFactor = 0.0
        self.setFont(QFont("Consolas"))

        self.filename_col = "filename"
        self.pred_col = "pred."
        self.tag_col = "tag"
        self.conf_score_col = "conf. score"
        self.origin_cols = [self.filename_col, self.pred_col, self.conf_score_col]
        self.tagged_cols = ["filename", self.pred_col, self.tag_col, self.conf_score_col]
        self.pred_col_fixed = f"{self.pred_col:<7}"        
        self.tag_col_fixed = f"{self.tag_col:<7}"
        self.conf_score_col_fixed = f"{self.conf_score_col:<7}"

        self.main_tag_true = "맞"
        self.main_tag_false = "틀"

        # self.sub_tag_none = ""
        # self.sub_tag_dict
        # self.sub_tag_err = "err"
        # self.sub_tag_diff = "diff"
        # self

        self.imageLabel = QLabel()

        # self.imageLabel.setBaseSize(500, 500)
        # self.imageLabel.resize(QSize(500, 500))
        self.imageLabel.setBackgroundRole(QPalette.Base)
        # self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.leftImageView = QScrollArea()
        self.leftImageView.setBackgroundRole(QPalette.Dark)
        self.leftImageView.setWidget(self.imageLabel)
        # self.leftImageView.setVisible(False)
        self.leftImageView.setVisible(True)
        
        # self.set_default_image_view()

        # radio_widget = QWidget(self)
        # radio_group = QButtonGroup(radio_widget)

        # # radio_0 = QRadioButton("0")
        # # radio_1 = QRadioButton("1")
        # # radio_2 = QRadioButton("2")
        # # radio_3 = QRadioButton("3")
        # radios = [QRadioButton(f"{i}") for i in range(3)]

        # radio_group = QButtonGroup()
        # for button in radios:
        #     radio_group.addButton(button)
        
        # self.tag_radio_widget = self.create_tag_radios()
        self.mainTagLabel = QLabel()
        self.mainTagLabel.setText("tag(main)")
        self.mainTagComboBox = self.createMainTagCombobox()
        self.mainTagComboBox.activated.connect(self.saveTagWithMainTag)

        self.subTagLabel = QLabel()
        self.subTagLabel.setText("tag(sub)")
        self.subTagLineEdit = self.createSubTagLineEdit()
        self.subTagLineEdit.returnPressed.connect(self.saveTagWithSubTag)
        # self.subTagComboBox = self.createSubTagCombobox()
        # self.subTagComboBox.activated.connect(self.saveTag)

        self.showPreviousButton = QPushButton("< Previous")
        self.showPreviousButton.clicked.connect(self.showPrevious)
        self.showNextButton = QPushButton("Next >")
        self.showNextButton.clicked.connect(self.showNext)

        self.central = QWidget(self)
        self.centralLayout = QHBoxLayout(self.central)
        self.leftSidebarLayout = QHBoxLayout(self.central)
        self.centerBoxLayout = QVBoxLayout(self.central)
        self.centerArrows = QHBoxLayout(self.central)
        self.centerDetails = QVBoxLayout(self.central)

        self.tagLayout = QHBoxLayout(self.central)
        self.mainTagLayout = QVBoxLayout(self.central)
        self.mainTagLayout.setAlignment(Qt.AlignVCenter)
        self.mainTagLayout.addWidget(self.mainTagLabel)
        self.mainTagLayout.addWidget(self.mainTagComboBox)
        self.subTagLayout = QVBoxLayout(self.central)
        self.subTagLayout.setAlignment(Qt.AlignVCenter)
        self.subTagLayout.addWidget(self.subTagLabel)
        self.subTagLayout.addWidget(self.subTagLineEdit)
        self.tagLayout.addLayout(self.mainTagLayout)
        self.tagLayout.addLayout(self.subTagLayout)

        self.predDetailLayout = QHBoxLayout(self.central)
        self.predLabel = QLabel()
        self.predLabel.setText(self.pred_col_fixed)
        self.predText = QLineEdit()
        self.predText.setEnabled(False)
        self.predDetailLayout.addWidget(self.predLabel)
        self.predDetailLayout.addWidget(self.predText)

        self.tagDetailLayout = QHBoxLayout(self.central)
        self.tagLabel = QLabel()
        self.tagLabel.setText(self.tag_col_fixed)
        self.tagText = QLineEdit()
        self.tagText.setEnabled(False)
        self.tagDetailLayout.addWidget(self.tagLabel)
        self.tagDetailLayout.addWidget(self.tagText)

        self.descDetailLayout = QHBoxLayout(self.central)
        self.descLabel = QLabel()
        self.descLabel.setText(self.conf_score_col_fixed)
        self.descText = QLineEdit()
        self.descText.setEnabled(False)
        self.descDetailLayout.addWidget(self.descLabel)
        self.descDetailLayout.addWidget(self.descText)

        self.centerDetails.addLayout(self.predDetailLayout)
        self.centerDetails.addLayout(self.tagDetailLayout)
        self.centerDetails.addLayout(self.descDetailLayout)
        
        self.rightSidebarLayout = QHBoxLayout(self.central)

        self.listWidget = QListWidget()
        self.listWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.get_image_list([])

        self.leftSidebarLayout.addWidget(self.listWidget)

        self.centralLayout.addLayout(self.leftSidebarLayout)
        self.centralLayout.addLayout(self.centerBoxLayout)
        self.centralLayout.addLayout(self.rightSidebarLayout)

        self.centerArrows.addWidget(self.showPreviousButton)
        self.centerArrows.addWidget(self.showNextButton)
        self.centerBoxLayout.addWidget(self.leftImageView)
        self.centerBoxLayout.addLayout(self.centerArrows)
        self.centerBoxLayout.addLayout(self.centerDetails)

        self.img_select_layout = QHBoxLayout(QWidget())
        
        self.rightSidebarLayout.addLayout(self.tagLayout)
        self.setCentralWidget(self.central)

        self.createActions()
        self.createMenus()

        # self.listWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.listWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.listWidget.itemSelectionChanged.connect(self.listOnSelection)
        self.listOnSelection()

        self.setWindowTitle("Image Viewer")
        self.resize(1200, 600)
    
    def createMainTagCombobox(self):
        combobox = QComboBox()
        combobox.addItem(self.main_tag_true)
        combobox.addItem(self.main_tag_false)
        
        return combobox

    def createSubTagLineEdit(self):
        lineEdit = QLineEdit()
        lineEdit.setText("")
        return lineEdit

    # def createSubTagCombobox(self):
    #     combobox = QComboBox()
    #     combobox.addItem("")
    #     combobox.addItem(self.sub_tag_err)
    #     combobox.addItem(self.sub_tag_diff)

    #     return combobox
    
    # def create_tag_radios(self):
    #     radio_widget = QWidget(self)
    #     radio_group = QButtonGroup(radio_widget)

    #     # radio_0 = QRadioButton("0")
    #     # radio_1 = QRadioButton("1")
    #     # radio_2 = QRadioButton("2")
    #     # radio_3 = QRadioButton("3")
    #     radios = [QRadioButton(f"{i}") for i in range(3)]

    #     radio_group = QButtonGroup()
    #     for button in radios:
    #         radio_group.addButton(button)
        
    #     return radio_widget

    @staticmethod
    def get_image_view_background() -> QImage:
        bg_image = np.ones((500, 500))
        h, w = bg_image.shape
        return QImage(bg_image.data, w, h, QImage.Format_BGR888)

    def set_default_image_view(self):
        self.imageLabel.setPixmap(
            QPixmap.fromImage(self.get_image_view_background())
        )
        self.scaleFactor = 1.0

        self.leftImageView.setVisible(True)
        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()

        self.imageLabel.adjustSize()
    
    def get_image_list(self, img_list):
        self.listWidget.clear()
        for i in img_list:
            item = QListWidgetItem()
            item.setText(i)
            self.listWidget.addItem(item)

    def get_idx_from_list(self, row: QListWidgetItem) -> int:
        idx = self.listWidget.indexFromItem(row)
        return idx.row()


    def listOnSelection(self):
        if self.listWidget.selectedItems():
            self.selected: QListWidgetItem = self.listWidget.selectedItems()[0]
            self.showImage(Path(self.selected.text()).as_posix())

            self.img_idx = self.get_idx_from_list(self.selected)

            self.predText.setText(f"{self.metadata.loc[self.img_idx, self.pred_col]}")
            self.tagText.setText(f"{self.metadata.loc[self.img_idx, self.tag_col]}")
            self.descText.setText(f"{self.metadata.loc[self.img_idx, self.conf_score_col]}")

        else:
            self.set_default_image_view()
            self.tagText.setText("")
            self.descText.setText("")

    def showPrevious(self):
        if self.listWidget.count() > 0:
            if self.listWidget.selectedItems():
                self.img_idx = self.get_idx_from_list(self.selected)
                if self.img_idx > 0:
                    self.listWidget.setCurrentRow(self.img_idx - 1)
            else:
                self.img_idx = 0
                self.listWidget.setCurrentRow(self.img_idx)

    def showNext(self):
        if self.listWidget.count() > 0:
            if self.listWidget.selectedItems():
                self.img_idx = self.get_idx_from_list(self.selected)
                if self.img_idx < self.listWidget.count() - 1:
                    self.listWidget.setCurrentRow(self.img_idx + 1)
            else:
                self.img_idx = 0
                self.listWidget.setCurrentRow(self.img_idx)


    def showImage(self, img_filename):
        image = QImage(img_filename)
        if image.isNull():
            QMessageBox.information(self, "Image Viewer", "Cannot load %s." % img_filename)
            return

        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.scaleFactor = 1.0

        self.leftImageView.setVisible(True)
        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()

        # self.imageLabel.setFixedSize(QSize(400, 400))
        self.imageLabel.adjustSize()
        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()


    def getSubTag(self):
        subTag = self.subTagLineEdit.text()
        # if self.subTagComboBox.currentText():
        #     subTag = self.subTagComboBox.currentText().title()
        # else:
        #     subTag = ""
        return subTag


    def saveTagWithMainTag(self):
        if self.listWidget.selectedItems():
            if self.mainTagComboBox.currentText():
                mainTag = self.mainTagComboBox.currentText().title()
                new_tag = f"{mainTag}{self.getSubTag()}"
                self.metadata.loc[self.img_idx, self.tag_col] = new_tag
                self.metadata.to_csv(self.filename, sep=self._sep, index=False, header=False)
                self.subTagLineEdit.setText("")
                # self.subTagLineEdit.selectAll()
                self.showNext()

    def saveTagWithSubTag(self):
        if self.listWidget.selectedItems():
            mainTag = self.main_tag_false
            new_tag = f"{mainTag}{self.getSubTag()}"
            self.metadata.loc[self.img_idx, self.tag_col] = new_tag
            self.metadata.to_csv(self.filename, sep=self._sep, index=False, header=False)
            self.subTagLineEdit.setText("")
            # self.subTagLineEdit.selectAll()
            self.showNext()

    def setTag(self, main_tag_str: str, sub_tag_str: str):
        if self.listWidget.selectedItems():
            self.mainTagComboBox.setCurrentIndex(
                self.mainTagComboBox.findText(main_tag_str)
            )
            # self.subTagLineEdit.text
            # self.subTagComboBox.setCurrentIndex(
            #     self.subTagComboBox.findText(sub_tag_str)
            # )
            self.saveTagWithMainTag()


    def setMainTagAsTrue(self):
        if self.listWidget.selectedItems():
            self.subTagLineEdit.setText("")
            self.mainTagComboBox.setCurrentIndex(
                self.mainTagComboBox.findText(self.main_tag_true)
            )
            # self.subTagComboBox.setCurrentIndex(
            #     self.mainTagComboBox.findText(self.sub_tag_none)
            # )
            self.saveTagWithMainTag()

    def setMainTagAsFalse(self):
        if self.listWidget.selectedItems():
            self.subTagLineEdit.setText("")
            self.mainTagComboBox.setCurrentIndex(
                self.mainTagComboBox.findText(self.main_tag_false)
            )
            # self.subTagComboBox.setCurrentIndex(
            #     self.mainTagComboBox.findText(self.sub_tag_none)
            # )
            self.saveTagWithMainTag()

    # def setSubTagAsErr(self):
    #     if self.listWidget.selectedItems():
    #         self.subTagComboBox.setCurrentIndex(
    #             self.mainTagComboBox.findText(self.sub_tag_err)
    #         )
    #         self.saveTag()

    # def setSubTagAsDiff(self):
    #     if self.listWidget.selectedItems():
    #         self.subTagComboBox.setCurrentIndex(
    #             self.mainTagComboBox.findText(self.sub_tag_diff)
    #         )
    #         self.saveTag()


    def open(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Text Files (*.csv *.tsv *.txt)', options=options)
        if filename:
            self.filename = filename
            if filename.endswith("csv"):
                self._sep = ","
                self.metadata = pd.read_csv(filename, sep=self._sep, header=None)
            else:
                self._sep = "\t"
                self.metadata = pd.read_csv(filename, sep=self._sep, header=None)

            if self.metadata.shape[1] == 3:
                self.metadata.columns = self.origin_cols
                self.metadata["tag"] = None
                self.metadata = self.metadata.loc[:, self.tagged_cols]
            elif self.metadata.shape[1] == 4:
                self.metadata.columns = self.tagged_cols
            else:
                raise ValueError("Data File Column shoud be one of {3, 4}")


            self.img_idx = 0

            ext_list = ["png", "jpg", "jpeg", "bmp", "gif"]
            self.get_image_list(self.metadata[self.filename_col].to_list())


    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.leftImageView.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QMessageBox.about(self, "About Image Viewer",
                          "<p>The <b>Image Viewer</b> example shows how to combine "
                          "QLabel and QScrollArea to display an image. QLabel is "
                          "typically used for displaying text, but it can also display "
                          "an image. QScrollArea provides a scrolling view around "
                          "another widget. If the child widget exceeds the size of the "
                          "frame, QScrollArea automatically provides scroll bars.</p>"
                          "<p>The example demonstrates how QLabel's ability to scale "
                          "its contents (QLabel.scaledContents), and QScrollArea's "
                          "ability to automatically resize its contents "
                          "(QScrollArea.widgetResizable), can be used to implement "
                          "zooming and scaling features.</p>"
                          "<p>In addition the example shows how to use QPainter to "
                          "print an image.</p>")

    def createActions(self):
        self.showPreviousAct = QAction("Show $Previous", self,
            shortcut="Ctrl+P",
            # shortcut=Qt.Key.Key_Left,
            triggered=self.showPrevious,
        )
        self.showNextAct = QAction("Show $Next", self,
            shortcut="Ctrl+N",
            # shortcut=Qt.Key.Key_Right,
            triggered=self.showNext,
        )
        self.openAct = QAction("&Open...", self, shortcut=QKeySequence("Ctrl+O"), triggered=self.open)
        self.printAct = QAction("&Print...", self, enabled=False, triggered=self.print_)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)
        self.fitToWindowAct = QAction("&Fit to Window", self, shortcut="Ctrl+W", enabled=False, checkable=True, triggered=self.fitToWindow)
        self.aboutAct = QAction("&About", self, triggered=self.about)
        self.aboutQtAct = QAction("About &Qt", self, triggered=qApp.aboutQt)
        self.mainTagAsTrueAct = QAction(f"Set Main Tag to '{self.main_tag_true}'", self, shortcut="Ctrl+Shift+T", triggered=self.setMainTagAsTrue)
        self.mainTagAsFalseAct = QAction(f"Set Main Tag to '{self.main_tag_false}'", self, shortcut="Ctrl+Shift+F", triggered=self.setMainTagAsFalse)

        # self.subTagAsErrAct = QAction(f"Set Main Tag to '{self.sub_tag_err}'", self, shortcut="Ctrl+E", triggered=self.setSubTagAsErr)
        # self.subTagAsDiffAct = QAction(f"Set Main Tag to '{self.sub_tag_diff}'", self, shortcut="Ctrl+D", triggered=self.setSubTagAsDiff)
        # self.mainFalsesubErrAct = QAction(f"Set Tag to '{self.main_tag_false}{self.sub_tag_err}'", self,
        #     shortcut=QKeySequence("Ctrl+Shift+E"),
        #     triggered=lambda : self.setTag(self.main_tag_false, self.sub_tag_err)
        # )
        # self.mainFalsesubDiffAct = QAction(f"Set Tag to '{self.main_tag_false}{self.sub_tag_diff}'", self,
        #     shortcut=QKeySequence("Ctrl+Shift+D"),
        #     triggered=lambda : self.setTag(self.main_tag_false, self.sub_tag_diff)
        # )

    def createMenus(self):
        self.fileMenu = QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.editMenu = QMenu("&Edit", self)
        self.editMenu.addAction(self.mainTagAsTrueAct)
        self.editMenu.addAction(self.mainTagAsFalseAct)
        # self.editMenu.addSeparator()
        # self.editMenu.addAction(self.subTagAsErrAct)
        # self.editMenu.addAction(self.subTagAsDiffAct)
        # self.editMenu.addAction(self.mainFalsesubErrAct)
        # self.editMenu.addAction(self.mainFalsesubDiffAct)

        self.viewMenu = QMenu("&View", self)
        self.viewMenu.addAction(self.showPreviousAct)
        self.viewMenu.addAction(self.showNextAct)

        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.editMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.leftImageView.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.leftImageView.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    imageViewer = QImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())
    # TODO QScrollArea support mouse
    # base on https://github.com/baoboa/pyqt5/blob/master/examples/widgets/imageviewer.py
    #
    # if you need Two Image Synchronous Scrolling in the window by PyQt5 and Python 3
    # please visit https://gist.github.com/acbetter/e7d0c600fdc0865f4b0ee05a17b858f2