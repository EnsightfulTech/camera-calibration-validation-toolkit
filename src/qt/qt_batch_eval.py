import os.path
import sys
from pathlib import Path
import numpy as np
import json
import cv2
import glob
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from tqdm import tqdm
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.eval_tools import find_chessboard_gray, reproject_mono_stereo

class EvaluationBatch(QWidget):
    def __init__(self):
        super(EvaluationBatch, self).__init__()
        self.initUI()
        self.save_dir = "src/test"

    def initUI(self):
        # 创建垂直布局
        vbox = QVBoxLayout()

        # 创建第一个选择文件的按钮
        self.file1_btn = QPushButton('choose camera model', self)
        self.file1_btn.clicked.connect(self.choose_file1)
        vbox.addWidget(self.file1_btn)

        # 创建第二个选择文件的按钮
        self.file2_btn = QPushButton('choose sbs_img folder', self)
        self.file2_btn.clicked.connect(self.choose_file2)
        vbox.addWidget(self.file2_btn)

        self.file3_btn = QPushButton('choose save directory', self)
        self.file3_btn.clicked.connect(self.choose_file3)
        vbox.addWidget(self.file3_btn)

        # 创建evaluation按钮
        self.eval_btn = QPushButton('evaluation', self)
        self.eval_btn.clicked.connect(self.evaluation)
        vbox.addWidget(self.eval_btn)

        # 创建水平布局
        hbox = QHBoxLayout()

        # 创建用于显示选择的文件路径的标签
        self.file1_label = QLabel('No camera model selected')
        hbox.addWidget(self.file1_label)

        self.file2_label = QLabel('No sbs_img folder selected')
        hbox.addWidget(self.file2_label)

        vbox.addLayout(hbox)

        self.setLayout(vbox)
        self.setGeometry(300, 300, 300, 150)
        self.show()

    def choose_file1(self):
        # 打开文件选择对话框
        filename = QFileDialog.getOpenFileName(self, '选择文件1', '.', 'JSON files (*.json)')
        # 更新文件路径标签
     
        if filename[0]!="":
            self.file1_label.setText(filename[0])
            with open(filename[0], "r") as read_file:
                decodedArray = json.load(read_file)
                try:
                    self.cm1 = np.asarray(decodedArray['cm1'])
                    self.cd1 = np.asarray(decodedArray['cd1'])
                    self.cm2 = np.asarray(decodedArray['cm2'])
                    self.cd2 = np.asarray(decodedArray['cd2'])
                    self.R = np.asarray(decodedArray['R'])
                    self.T = np.asarray(decodedArray['T'])
                    self.image_size =  np.asarray(decodedArray['image_size'])
                    
                except: # if encounter None object, then no assignment
                    pass

    def choose_file2(self):
        # 打开文件选择对话框
        dirname = QFileDialog.getExistingDirectory(None, "选择文件夹")
        if dirname:    
            self.imgs_dir = dirname
            self.file2_label.setText(dirname)

    def choose_file3(self):
        default_path = "src/test"
        dirname = QFileDialog.getExistingDirectory(None, "选择文件夹", default_path)
        if dirname:    
            self.save_dir = dirname

    def evaluation(self):
        print("read images...")

        images = glob.glob(self.imgs_dir+"/*.jpg")
        left_error_mono = []
        right_error_mono = []
        right_error_stereo = []
        imgpoints_left = []
        imgpoints_right = []
        for fname in tqdm(images):
            img= cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grayL = gray [:,                    0:  gray.shape[1]//2]
            grayR = gray [:, gray.shape[1]//2:  gray.shape[1]]
            cornersL = find_chessboard_gray(grayL)
            cornersR = find_chessboard_gray(grayR)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

        # imgpoints_left = np.load("src/test/points_left.npy")
        # imgpoints_right = np.load("src/test/points_right.npy")
        print("calculating errors")
        for i in range(len(imgpoints_left)):
            proj_points_L, proj_points_R,proj_pointsR_stereo = reproject_mono_stereo(imgpoints_left[i],imgpoints_right[i],self.cm1,self.cd1,self.cm2,self.cd2,self.R,self.T)
            
            errorL = cv2.norm(imgpoints_left[i],proj_points_L, cv2.NORM_L2)
            errorL = errorL / (len(proj_points_L)**0.5)
            left_error_mono.append(errorL)

            errorR = cv2.norm(imgpoints_right[i],proj_points_R, cv2.NORM_L2) 
            errorR = errorR / (len(proj_points_R)**0.5)
            right_error_mono.append(errorR)

            errorR_stereo = cv2.norm(imgpoints_right[i],proj_pointsR_stereo, cv2.NORM_L2) 
            errorR_stereo = errorR_stereo / (len(proj_pointsR_stereo)**0.5)
            right_error_stereo.append(errorR_stereo)

        rpe_L = sum(left_error_mono)/len(left_error_mono)
        rpe_R = sum(right_error_mono)/len(right_error_mono)
        rpe_R_stereo =sum(right_error_stereo)/len(right_error_stereo)
        info = f'''left mono reprojection error: mean:{rpe_L} min:{min(left_error_mono)} max:{max(left_error_mono)}\n right mono reprojection error: mean:{rpe_R} min:{min(right_error_mono)} max{max(right_error_mono)}\nright stereo reprojection error: mean:{rpe_R_stereo} min:{min(right_error_stereo)} max:{max(right_error_stereo)}
        '''
        print(info)
        txt_dir = self.save_dir+"/"+self.imgs_dir.split("/")[-1]+".txt"
        file = open(txt_dir, 'w', encoding='utf-8')
        file.write(info)
        file.write("\n\n")
        detail_info = zip(images,left_error_mono,right_error_mono,right_error_stereo)
        file.write("\n".join([str(item) for item in detail_info]))
        print("done")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = EvaluationBatch()
    gui.show()
    sys.exit(app.exec())