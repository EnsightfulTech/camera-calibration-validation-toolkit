import numpy as np
import cv2
from statistics import mean, stdev

CHECKERBOARD=(11, 8)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.000001)

def find_chessboard(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),SUBPIX_CRITERIA)
    return corners
def find_chessboard_gray(gray):
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),SUBPIX_CRITERIA)
    return corners


def reproject(corners,cm,cd):
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
    # retval, rvecs, tvecs = cv2.solvePnP(objp, corners, cm, cd)
    retval, rvecs, tvecs = cv2.solvePnP(objp, corners, cm, cd)
    proj_points,_ = cv2.projectPoints(objp, rvecs, tvecs, cm, cd)
    return proj_points

def reproject_stereo(cornersL, cornersR,corners_rect_L,corners_rect_R, Q, cm1, cd1, cm2, cd2, R, T):
    # objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
    # for i in range(0, 88):
    #     img_coord = [corners_rect_L[i][0], corners_rect_R[i][0]]
    #     coord = get_world_coord_Q(Q, img_coord[0], img_coord[1],0)
    #     objp[i,:] = np.array(coord)

    # min_x,min_y,_ = objp.min(axis=0)
    # delta_x = -min_x if min_x<0 else 0
    # delta_y = -min_y if min_y<0 else 0
    # objp = objp + np.array([delta_x,delta_y,0])   

    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
    objp =objp*60 
 
    retL, rvecsL, tvecsL = cv2.solvePnP(objp, cornersL, cm1, cd1)
    #retR, rvecsR, tvecsR = cv2.solvePnP(objp, cornersR, cm2, cd2)
    
    proj_pointsL,_ = cv2.projectPoints(objp, rvecsL, tvecsL, cm1, cd1)
    # proj_pointsR,_ = cv2.projectPoints(objp, rvecsR, tvecsR, cm2, cd2)

    rvecsR,tvecsR = cv2.composeRT(rvecsL, tvecsL,cv2.Rodrigues(R)[0],T)[:2]

    proj_pointsR,_ = cv2.projectPoints(objp, rvecsR, tvecsR, cm2, cd2) 

    return proj_pointsL.astype(np.float32), proj_pointsR.astype(np.float32)

def reproject_mono_stereo(cornersL, cornersR, cm1, cd1, cm2, cd2, R, T):
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
    objp =objp*60 

    retL, rvecsL, tvecsL = cv2.solvePnP(objp, cornersL, cm1, cd1)
    retR, rvecsR, tvecsR = cv2.solvePnP(objp, cornersR, cm2, cd2)
    proj_pointsL,_ = cv2.projectPoints(objp, rvecsL, tvecsL, cm1, cd1)
    proj_pointsR,_ = cv2.projectPoints(objp, rvecsR, tvecsR, cm2, cd2) 

    rvecsR_stereo,tvecsR_stereo = cv2.composeRT(rvecsL, tvecsL,cv2.Rodrigues(R)[0],T)[:2]
    proj_pointsR_stereo,_ = cv2.projectPoints(objp, rvecsR_stereo, tvecsR_stereo, cm2, cd2) 

    return proj_pointsL.astype(np.float32), proj_pointsR.astype(np.float32),proj_pointsR_stereo.astype(np.float32)

def reproject_error(detect_pts,proj_pts):
    error =  cv2.norm(detect_pts,proj_pts, cv2.NORM_L2)
    error = error / (len(proj_pts)**0.5)
    return error

def rectify(img, cm, cd, R, P, newImageSize):
    MapX, MapY  = cv2.initUndistortRectifyMap(cm, cd, R, P, newImageSize, cv2.CV_16SC2)
    rect = cv2.remap(img, MapX, MapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rect
    
def calculateRT(cornersL,cornersR,cm1,cd1,cm2,cd2):
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
    objp =objp*60 

    retL, rvecsL, tvecsL = cv2.solvePnP(objp, cornersL, cm1, cd1)
    retR, rvecsR, tvecsR = cv2.solvePnP(objp, cornersR, cm2, cd2)

    rAO = cv2.Rodrigues(rvecsL)[0].T
    tAO= -np.dot(rAO, tvecsL.reshape(3,))
    r,t = cv2.composeRT(cv2.Rodrigues(rAO)[0], tAO.reshape(3,1),rvecsR, tvecsR)[:2]

    return r,t
def convert_angle(R):
    y = np.arctan2(-R[2][0],np.sqrt(R[0][0]**2+R[1,0]**2))
    x = np.arctan2(R[1][0]/np.cos(y),R[0][0]/np.cos(y))
    z = np.arctan2(R[2][1]/np.cos(y),R[2][2]/np.cos(y))
    
    return np.rad2deg(x),np.rad2deg(y),np.rad2deg(z)
def eval_box_edge_len(cornersL, cornersR, Q):
    # box edge lengths
    edges = []
    for i in range(CHECKERBOARD[0] * CHECKERBOARD[1] - 1):
        # select two points
        point_id_1 = i
        point_id_2 = i+1

        img_coord_1 = [cornersL[point_id_1][0], cornersR[point_id_1][0]]
        img_coord_2 = [cornersL[point_id_2][0], cornersR[point_id_2][0]]

        coord1 = get_world_coord_Q(Q, img_coord_1[0], img_coord_1[1],0)
        coord2 = get_world_coord_Q(Q, img_coord_2[0], img_coord_2[1],0)

        distance = cv2.norm(coord1, coord2)
        if distance < 100:
            edges.append(distance)
        else:
            # print(point_id_1, point_id_2)
            pass
        
    if len(edges)>=2:
        msg = f'''Box Length Performance:
                Max: {max(edges)}; Min: {min(edges)}
                Mean: {mean(edges)}; Stdev: {stdev(edges)} 
                '''
    else:
        msg = "No correct box edge detected"
    return msg


def eval_long_edge_len( cornersL, cornersR, Q):
    # long edge lengths
    edges = []
    for i in range(0, 87, 11):
        # select two points
        point_id_1 = i
        point_id_2 = i+10

        img_coord_1 = [cornersL[point_id_1][0], cornersR[point_id_1][0]]
        img_coord_2 = [cornersL[point_id_2][0], cornersR[point_id_2][0]]

        coord1 = get_world_coord_Q(Q, img_coord_1[0], img_coord_1[1],0)
        coord2 = get_world_coord_Q(Q, img_coord_2[0], img_coord_2[1],0)
        distance = cv2.norm(coord1, coord2)
        edges.append(distance)
        
    msg = f'''Long Edge Length Performance:
        Max: {max(edges)}; Min: {min(edges)}
        Mean: {mean(edges)}; Stdev: {stdev(edges)}
        '''
    return msg


def get_world_coord_Q(Q, img_coord_left, img_coord_right,d):
    """Compute world coordniate by the Q matrix
    
    img_coord_left:  segment endpoint coordinate on the  left image
    img_coord_right: segment endpoint coordinate on the right image
    """
    x, y = img_coord_left
    disp = img_coord_left[0] - img_coord_right[0]
    # print("dispairty from map", d)
    # print("disp by subtraction",disp)
    # print(x, y, d); exit(0)
    homg_coord = Q.dot(np.array([x, y, disp, 1.0]))
    coord = homg_coord / homg_coord[3]
    # print(coord[:-1])
    return coord[:-1]

                                                                                                                                        
#CHECKERBOARD put where?