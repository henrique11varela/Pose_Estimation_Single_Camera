import cv2 as cv
import numpy as np
from calibrate import calibrate
import math
from sympy import symbols, Eq, solve

def find_perpendicular_vectors(x1, y1, x2, y2, r):
    # Define symbols for z1 and z2
    z1, z2 = symbols('z1 z2')
    
    # 1. Perpendicularity condition: v1 . v2 = 0
    # x1 * x2 + y1 * y2 + z1 * z2 = 0
    eq1 = Eq(x1 * x2 + y1 * y2 + z1 * z2, 0)
    
    # 2. Length ratio condition: ||v1|| / ||v2|| = r
    # (x1^2 + y1^2 + z1^2) = r^2 * (x2^2 + y2^2 + z2^2)
    eq2 = Eq(x1**2 + y1**2 + z1**2, r**2 * (x2**2 + y2**2 + z2**2))
    
    # Solve the system of equations
    solution = solve((eq1, eq2), (z1, z2))
    
    return [(x1, y1, solution[0][0]),(x2, y2, solution[0][1])]

def calculate_length(p1):
        p1v = p1[0]**2 + p1[1]**2 + p1[2]**2
        return math.sqrt(p1v)

def find_point_at_y(y, v):
    x0, y0, z0 = (0, 0, 0)  # Known point on the line
    vx, vy, vz = v   # Direction vector
    
    # Check if the direction vector has a non-zero y-component
    if vy == 0:
        raise ValueError("The direction vector is parallel to the xz-plane; it never reaches the specified y-coordinate.")
    
    # Calculate the parameter t for the specified y-coordinate
    t = (y - y0) / vy
    
    # Calculate the corresponding x and z coordinates
    x = x0 + t * vx
    z = z0 + t * vz
    
    return (x, y, z)

if __name__ == '__main__':
    # camMatrix is the camera information like focal length and center of image
    # distCoeff is the lens distortion information
    camMatrix, distCoeff = calibrate(showPics=True)

    C = (687, 980)
    A = (447, 278)
    B = (632, 379)
    F = (1398, 270)

    print('')
    print('2D uv-------------------------')
    print(f'C{C}')
    print(f'A{A}')
    print(f'B{B}')
    print(f'F{F}')

    undistorted_points = cv.undistortPoints(np.array([[C], [A], [B], [F]], dtype=np.float32), camMatrix, distCoeff)
    pixel_points = np.empty_like(undistorted_points)
    fx = camMatrix[0][0]
    fy = camMatrix[1][1]
    cx = camMatrix[0][2]
    cy = camMatrix[1][2]
    for i, (xn, yn) in enumerate(undistorted_points.reshape(-1, 2)):
        x_pixel = fx * xn + cx
        y_pixel = fy * yn + cy
        pixel_points[i] = [[x_pixel, y_pixel]]
    
    Cuv = pixel_points[0][0]
    Auv = pixel_points[1][0]
    Buv = pixel_points[2][0]
    Fuv = pixel_points[3][0]

    print('')
    print('2D undistorted uv-------------------------')
    print(f'Cuv{Cuv}')
    print(f'Auv{Auv}')
    print(f'Buv{Buv}')
    print(f'Fuv{Fuv}')

    projC = (Cuv[0] - Buv[0], -(Cuv[1] - Buv[1]))
    projA = (Auv[0] - Buv[0], -(Auv[1] - Buv[1]))
    projB = (Buv[0] - Buv[0], -(Buv[1] - Buv[1]))
    projF = (Fuv[0] - Buv[0], -(Fuv[1] - Buv[1]))

    print('')
    print('2D projected to B(0,0,0) uv-------------------------')
    print(f'C{projC}')
    print(f'A{projA}')
    print(f'B{projB}')
    print(f'F{projF}')

    A3d, F3d = find_perpendicular_vectors(projA[0], projA[1], projF[0], projF[1], 80/155)
    c = np.cross(A3d, F3d)

    print('')
    print(f'VECTOR: {c}')
    print('')

    C3d = find_point_at_y(projC[1], c)

    print('')
    print('3D projected to B(0,0,0) uv-------------------------')
    print(f'A={A3d}')
    print(f'F={F3d}')
    print(f'C={C3d}')
    print('')
    print(f'A.F: {np.dot(A3d, F3d)}')
    print(f'A.C: {np.dot(A3d, C3d)}')
    print(f'C.F: {np.dot(C3d, F3d)}')

    Al = calculate_length(A3d)
    Fl = calculate_length(F3d)
    Cl = calculate_length(C3d)

    print('')
    print('lengths-------------------------')
    print(f'Al: {Al}px')
    print(f'Fl: {Fl}px')
    print(f'Cl: {Cl}px')
    
    print('')
    print('ratios real/projected-------------------------')
    print(f'{80 / Al}')
    print(f'{155 / Fl}')
    print(f'{95 / Cl}')

    print('')
    print(f'real ratio: {155 / 80}')
    print(f'calculated ratio: {Fl / Al}')

    estimated_length = (155 / Fl) * Cl

    print('')
    print(f'Estimated length: {estimated_length:.2f}mm')


    