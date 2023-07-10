import os.path
import sys
from pathlib import Path

import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.eval_tools import find_chessboard_charuco, reproject, rectify,eval_box_edge_len, eval_long_edge_len, reproject_stereo,calculateRT,convert_angle


def draw(img, subpix_corners, proj_points):
    def on_scroll(event):
        # 获取当前的x轴和y轴的缩放比例
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

        direction = event.button
        if direction == 'down':
            scale_factor = 1 / 4
        elif direction == 'up':
            scale_factor = 4
        else:
            scale_factor = 1

        # 获取鼠标指针的位置
        x, y = event.xdata, event.ydata

        # 计算新的x轴和y轴的缩放比例
        new_xlim = [(cur_xlim[0] - x) * scale_factor + x, (cur_xlim[1] - x) * scale_factor + x]
        new_ylim = [(cur_ylim[0] - y) * scale_factor + y, (cur_ylim[1] - y) * scale_factor + y]

        # 设置新的x轴和y轴的缩放比例
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

        # 重新绘制图像
        plt.draw()

    # 创建一个图像窗口
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.scatter(subpix_corners[:, 0, 0], subpix_corners[:, 0, 1], color='r', s=1)
    ax.scatter(proj_points[:, 0, 0], proj_points[:, 0, 1], color='y', s=1)
    # 绑定鼠标滚轮事件处理器
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    plt.show()

class Evaluation(QWidget):
    def __init__(self):
        super(Evaluation, self).__init__()
        self.initUI()
        self.save_dir = "src/test"
        # with open("src/test/camera_model.json", "r") as read_file:
        #     decodedArray = json.load(read_file)
        #     try:
        #         self.cm1 = np.asarray(decodedArray['cm1'])
        #         self.cd1 = np.asarray(decodedArray['cd1'])
        #         self.cm2 = np.asarray(decodedArray['cm2'])
        #         self.cd2 = np.asarray(decodedArray['cd2'])
        #         self.R = np.asarray(decodedArray['R'])
        #         self.T = np.asarray(decodedArray['T'])
        #         self.image_size =  np.asarray(decodedArray['image_size'])
        #     except: # if encounter None object, then no assignment
        #         pass
        # self.image_path = "/home/hyx/Downloads/Feishu20230601-111436.jpg"

    def initUI(self):
        # 创建垂直布局
        vbox = QVBoxLayout()

        # 创建第一个选择文件的按钮
        self.file1_btn = QPushButton('choose camera model', self)
        self.file1_btn.clicked.connect(self.choose_file1)
        vbox.addWidget(self.file1_btn)

        # 创建第二个选择文件的按钮
        self.file2_btn = QPushButton('choose sbs_img', self)
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

        self.file2_label = QLabel('No sbs_img selected')
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
        filename = QFileDialog.getOpenFileName(self, '选择文件2', '.', 'JPG files (*.jpg)')
        # 更新文件路径标签
        if filename[0]!="":
            self.file2_label.setText(filename[0])
            self.image_path = filename[0] 

    def choose_file3(self):
        default_path = "src/test"
        dirname = QFileDialog.getExistingDirectory(None, "选择文件夹", default_path)
        if dirname:    
            self.save_dir = dirname

    def evaluation(self):
        sbs_img = cv2.imread(str(self.image_path))
        assert sbs_img is not None
        img_left = sbs_img [:,                    0:  sbs_img.shape[1]//2]
        img_right = sbs_img [:, sbs_img.shape[1]//2:  sbs_img.shape[1]]

        cornersL,cornersIdL = find_chessboard_charuco(img_left)
        cornersR,cornersIdR = find_chessboard_charuco(img_right)
        proj_points_L_ori = reproject(cornersL, self.cm1, self.cd1)
        errorL_ori = cv2.norm(cornersL,proj_points_L_ori, cv2.NORM_L2)
        errorL_ori= errorL_ori / (len(proj_points_L_ori)**0.5)
        
        proj_points_R_ori = reproject(cornersR, self.cm2, self.cd2)
        errorR_ori = cv2.norm(cornersR,proj_points_R_ori, cv2.NORM_L2)
        errorR_ori= errorR_ori / (len(proj_points_R_ori)**0.5)
        

        errorL_msg_mono = "Left image mono reprojection error: {}\n".format(errorL_ori)
        errorR_msg_mono = "Right image mono reprojection error: {}\n".format(errorR_ori)
        print(errorL_msg_mono)
        print(errorR_msg_mono)


        R1, R2, P1, P2, Q, ROI1, ROI2 = \
            cv2.stereoRectify(
                self.cm1, self.cd1, 
                self.cm2, self.cd2, 
                self.image_size, 
                self.R, self.T,
                alpha=1,
                newImageSize=self.image_size,
            )
        rectL = rectify(img_left, self.cm1, self.cd1, R1, P1, self.image_size)
        rectR = rectify(img_right, self.cm2, self.cd2, R2, P2, self.image_size)
        corners_rect_L,_ = find_chessboard_charuco(rectL)
        corners_rect_R,_ = find_chessboard_charuco(rectR)

        #test new algo

        proj_points_L, proj_points_R = reproject_stereo(cornersL, self.cm1,self.cd1, self.cm2, self.cd2,self.R,self.T)

        errorL = cv2.norm(cornersL,proj_points_L, cv2.NORM_L2)
        errorL = errorL / (len(proj_points_L)**0.5)

        errorR = cv2.norm(cornersR,proj_points_R, cv2.NORM_L2) 
        errorR = errorR / (len(proj_points_R)**0.5)

        file_name = os.path.basename(self.image_path)
        txt_dir = self.save_dir+"/"+file_name[:-4]+".txt"
        file = open(txt_dir, 'w', encoding='utf-8')
        errorL_msg = "Left image stereo reprojection error: {}\n".format(errorL)
        errorR_msg = "Right image stereo reprojection error: {}\n".format(errorR)
        file.write(errorL_msg_mono)
        file.write(errorR_msg_mono)
        file.write(errorL_msg)
        file.write(errorR_msg+"\n")
        print(errorL_msg)
        print(errorR_msg)

        draw(img_left, cornersL, proj_points_L)
        draw(img_right,cornersR, proj_points_R)


        cal_r, cal_t = calculateRT(cornersL,cornersR,self.cm1,self.cd1,self.cm2,self.cd2)
        ori_x, ori_y,ori_z=convert_angle(self.R)
        x,y,z = convert_angle(cv2.Rodrigues(cal_r)[0])
        RTmessage = f'''The original rotation angle: 
        x:{ori_x},y:{ori_y},z:{ori_z}
        The calculated rotation angle: x:{x},y:{y},z:{z}
        Difference: x:{ori_x-x},y:{ori_y-y},z:{ori_z-z}
        The original translation vector:{self.T}
        The calculated translation vector:{cal_t.T} 
        '''
        print(RTmessage)
        file.write(RTmessage+"\n")

        box_len_msg =eval_box_edge_len(corners_rect_L, corners_rect_R, Q)
        box_long_msg =eval_long_edge_len(corners_rect_L, corners_rect_R, Q)
        file.write(box_len_msg+"\n")
        file.write(box_long_msg)
        print(box_len_msg)
        print(box_long_msg)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = Evaluation()
    gui.show()
    sys.exit(app.exec())