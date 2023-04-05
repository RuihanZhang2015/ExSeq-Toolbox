from PyQt5 import uic
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPixmap, QImage, QTextCursor
from PyQt5.QtWidgets import QApplication, QPushButton, QHBoxLayout, QWidget, QLabel, QSlider, QSpacerItem, QSizePolicy, \
    QLineEdit,QShortcut
import numpy as np
from scipy.spatial.transform import Rotation
from os import path
from sklearn.neighbors import NearestNeighbors
from skimage.draw import line_aa
from math import isnan
import traceback


# This is my local path, I copied these files in /mp/nas3/fixstars/yves/zebrafish_data/20221017_alignment/icp
WORKDIR = "/home/yves/Private/Fixstars/datasets/ZebrafishBrain/20221017_alignment/"

POINTS1_PATH = f"{WORKDIR}/icp/slice1_centers.npy"
POINTS2_PATH = f"{WORKDIR}/icp/slice2_centers.npy"


def getLineAtPosition(w):
  cursor = w.textCursor()
  pos = cursor.position()
  cursor.setPosition(pos)

  cursor.movePosition(QTextCursor.StartOfLine)
  lines = 0

  lines_text = cursor.block().text().splitlines()
  lines_pos = 0
  for line_text in lines_text:
    lines_pos += len(line_text) + 1
    if lines_pos > cursor.position() - cursor.block().position():
      break
    lines += 1

  block = cursor.block().previous()
  while block.isValid():
    lines += block.lineCount()
    block = block.previous()

  return lines

def plot_on(pts, dest, color=(255,255,255)):
    # Remove points that have negative coordinates
    p = pts[np.sum(pts>0, 1) == 2]
    # Remove points that have coordinates bigger than dest shape
    p=p[np.sum(p<np.array(dest.shape)[[1,0]], 1) == 2].astype(np.int32)
    dest[p[:, 1], p[:, 0], :] = np.array(color)
    return dest

def filter_points(pts, z_min=0, z_max=-1):
    p = pts[pts[:,2]>z_min]
    p = p[p[:,2]<z_max]
    return p


def neigh(p1, p2):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(p1)
    distances, indices = nbrs.kneighbors(p2)
    vec = np.mean(p2 - p1[indices[:, 0]], axis=0)
    return vec, np.mean(distances), indices


# Draws an AA line
def line(img, x1, y1, x2, y2, color):
    rr, cc, val = line_aa(int(y1), int(x1), int(y2), int(x2))
    img[rr, cc, 0] = val * color[0]
    img[rr, cc, 1] = val * color[1]
    img[rr, cc, 2] = val * color[2]
    return img


def plot_points(pts):
    p=pts.astype(np.int64)
    dest = np.zeros((np.max(p, axis=0)+1).astype(np.uint))
    dest[p[:,0], p[:,1], p[:,2]]=255
    return dest



def transform(p, trans, rot, scale=0.2, dest=None):
    p = (scale*p)
    r=Rotation.from_euler('z', rot, degrees=True)
    p=np.matmul(r.as_matrix(), p.T).T
    # p=p-np.min(p, axis=0)
    p=p+np.array([200,200,0])
    p=p+trans
    return p

def toPixMap(arr):
    if len(arr.shape) == 2:
        arr = np.stack((arr,arr,arr), axis=2)
    height, width, channel = arr.shape
    bytesPerLine = 3 * width
    qImg = QImage(arr.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)

class App(QApplication):
    def __init__(self):
        super().__init__([])
        Form, Window = uic.loadUiType("GUI.ui")
        self.window = Window()

        self.form = Form()
        self.form.setupUi(self.window)

        self.form.sliderX.valueChanged.connect(self.sliders_changed)
        self.form.sliderY.valueChanged.connect(self.sliders_changed)
        self.form.sliderR.valueChanged.connect(self.sliders_changed)
        self.form.sliderZ1.valueChanged.connect(self.sliders_changed)
        self.form.sliderZ2.valueChanged.connect(self.sliders_changed)
        self.form.loadButton.clicked.connect(self.load_images)
        self.form.runOnceButton.clicked.connect(self.run_once_button)
        self.form.clearButton.clicked.connect(self.form.logText.clear)

        self.form.runOnceButton.setShortcut("Ctrl+Return")
        # self.form.sliderX.setShortcut("Ctrl+Up")

        self.form.LblImage.mousePressEvent = self.on_click

        self.lastWindowClosed.connect(self.on_close)

        self.last_click_x = 145
        self.last_click_y = 151

        if path.exists("saved_code_autorun.py"):
            self.form.codeAutorun.setPlainText(open("saved_code_autorun.py", "r").read())
        if path.exists("saved_code_onclick.py"):
            self.form.codeOnClick.setPlainText(open("saved_code_onclick.py", "r").read())

        self.window.show()

        # Set starting values
        self.form.sliderX.setValue(5)
        self.form.sliderY.setValue(8)
        self.form.sliderZ1.setValue(30)
        self.form.sliderZ2.setValue(40)

        self.form.Xvalue.setText(str(self.form.sliderX.value()))
        self.form.Yvalue.setText(str(self.form.sliderY.value()))
        self.form.Rvalue.setText(str(self.form.sliderR.value()))
        self.form.Z1value.setText(str(self.form.sliderZ1.value()))
        self.form.Z2value.setText(str(self.form.sliderZ2.value()))

        self.form.showGrids.clicked.connect(self.sliders_changed)
        self.form.showOrigP1.clicked.connect(self.sliders_changed)
        self.form.showOrigP2.clicked.connect(self.sliders_changed)
        self.form.showDefPoints.clicked.connect(self.sliders_changed)
        self.form.showRaw.clicked.connect(self.sliders_changed)

        # QShortcut("Ctrl+Up", self.form.codeOnClick, self.up_key)
        # QShortcut(Qt.Key_Down, self, self.down_key)
        # QShortcut(Qt.Key_Left, self, self.left_key)
        # QShortcut(Qt.Key_Right, self, self.right_key)

        self.load_images()
        try:
            self.run_once_button()
        except Exception:
            print(traceback.format_exc())

        QApplication.instance().installEventFilter(self)

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress:
            self.form.lineCurrent.setText(f"Line: {getLineAtPosition(self.form.codeAutorun)}")
            if type(source)==QLabel:
                if event.key() == Qt.Key_Left or event.key() == Qt.Key_A:
                    self.form.sliderY.setValue(self.form.sliderY.value()+1)
                elif event.key() == Qt.Key_Right or event.key() == Qt.Key_D:
                    self.form.sliderY.setValue(self.form.sliderY.value() - 1)
                elif event.key() == Qt.Key_Up or event.key() == Qt.Key_W:
                    self.form.sliderX.setValue(self.form.sliderX.value() - 1)
                elif event.key() == Qt.Key_Down or event.key() == Qt.Key_S:
                    self.form.sliderX.setValue(self.form.sliderX.value() + 1)
        return super().eventFilter(source, event)


    def on_click(self, evt):
        # ctxt = dict(globals(),
        #             **{'pt1': self.pt1,
        #                'pt2': self.pt2,
        #                'self': self})
        codes = self.form.codeOnClick.toPlainText().split("###STOP")
        print(len(codes))
        exec(codes[0], globals(), locals())

    def on_close(self):
        f = open("saved_code_autorun.py","w")
        f.write(self.form.codeAutorun.toPlainText())
        f.close()
        f = open("saved_code_onclick.py","w")
        f.write(self.form.codeOnClick.toPlainText())
        f.close()

    def load_images(self):
        self.pt1 = np.load(POINTS1_PATH)
        self.pt2 = np.load(POINTS2_PATH)

    def run_once_button(self):
        self.run_code()
        self.update_images()

    def run_code(self):
        ctxt = dict(globals(),
                    **{'pt1': self.pt1,
                       'pt2': self.pt2,
                       'self': self})

        codes = self.form.codeAutorun.toPlainText().split("###STOP")
        print(len(codes))
        exec(codes[0], globals(), locals())

    def sliders_changed(self):
        self.form.Xvalue.setText(str(self.form.sliderX.value()))
        self.form.Yvalue.setText(str(self.form.sliderY.value()))
        self.form.Rvalue.setText(str(self.form.sliderR.value()))
        self.form.Z1value.setText(str(self.form.sliderZ1.value()))
        self.form.Z2value.setText(str(self.form.sliderZ2.value()))


        if self.form.runOnUpdatesButton.isChecked():
            self.run_code()
        self.update_images()

    def update_images(self):
        self.form.LblImage.setPixmap(toPixMap(self.outimg))
        self.form.LblImageSide.setPixmap(toPixMap(self.outimg2))


    def log(self, s):
        self.form.logText.appendPlainText(str(s))
        #self.form.logText.setPlainText(s)



app = App()
app.exec_()

