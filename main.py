from calibrate import calibrate
from poseEstimation import poseEstimation, DrawOption

if __name__ == '__main__':
    camMatrix, distCoeff = calibrate(showPics=False)
    poseEstimation(DrawOption.CUBE, camMatrix, distCoeff)

