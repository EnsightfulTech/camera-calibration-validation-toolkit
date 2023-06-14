import cv2
import glob
import numpy as np
from tqdm import tqdm
from src.utils.json_tools import dump_stereo_model
CHARUCO_CHECKERBOARD = (12,9) #charuco need to add 1 in each dimension
CHECKERBOARD = (11,8)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard(CHARUCO_CHECKERBOARD, .06, .045, dictionary)#.045m
w,h = 4032,3040
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
stereocalibration_flags = cv2.CALIB_USE_INTRINSIC_GUESS

demo_path="/home/hyx/Desktop/0609charuco/*.jpg" 
save_path = "camera_model_?.json"

leftCorners = []
leftIds = []
rightCorners = []
rightIds = []

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp = objp*60


objpoints = []
commonCornersL = []
commonCornersR = []

for fname in tqdm(glob.glob(demo_path)): 
    im=cv2.imread(fname,0)
    left = im[:, 0: w]
    right = im[:, w:]
    charucoDetector = cv2.aruco.CharucoDetector(board)
    charucoCornersL, charucoIdsL,_,_ = charucoDetector.detectBoard(left)
    charucoCornersR, charucoIdsR,_,_ = charucoDetector.detectBoard(right)
    if charucoIdsL is not None:
        leftCorners.append(charucoCornersL)
        leftIds.append(charucoIdsL)
    if charucoIdsR is not None:
        rightCorners.append(charucoCornersR)
        rightIds.append(charucoIdsR)
        
    commonL = []
    commonR = []
    objps = []
    d = {}  
    if charucoIdsL is not None and charucoIdsR is not None:
        for i in range(len(charucoIdsL)):
            id =  charucoIdsL[i][0]
            d[id] = charucoCornersL[i]
        for i in range(len(charucoIdsR)):
            id =  charucoIdsR[i][0]  
            if id in d:                           
                commonL.append(d[id])
                commonR.append(charucoCornersR[i])
                objps.append(objp[id])             
        if len(objps) >8: 
            commonCornersL.append(np.array(commonL))
            commonCornersR.append(np.array(commonR))
            objpoints.append(np.array(objps))

_, cameraMatrixL, distCoeffsL, rvecsL, tvecsL, stdDeviationsIntrinsicsL, stdDeviationsExtrinsicsL, perViewErrorsL = cv2.aruco.calibrateCameraCharucoExtended(leftCorners, leftIds, board,(w,h),None,None)
_, cameraMatrixR, distCoeffsR, rvecsR, tvecsR, stdDeviationsIntrinsicsR, stdDeviationsExtrinsicsR, perViewErrorsR = cv2.aruco.calibrateCameraCharucoExtended(rightCorners, rightIds, board,(w,h),None,None)

ret, CM1, dist1, CM2, dist2, R, T, E, F, rvecs, tvecs, perViewErrors=cv2.stereoCalibrateExtended(objpoints,commonCornersL,commonCornersR,\
    cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR,(w,h),R=None,T=None,criteria=criteria,flags=stereocalibration_flags)

dump_stereo_model( CM1, dist1, CM2, dist2, R, T,save_path)

# print(perViewErrors[0])