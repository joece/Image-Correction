import numpy as np
import cv2

def CrossPoint(line1, line2): 
    x0, y0, x1, y1 = line1[0]
    x2, y2, x3, y3 = line2[0]

    dx1 = x1 - x0
    dy1 = y1 - y0

    dx2 = x3 - x2
    dy2 = y3 - y2
    
    D1 = x1 * y0 - x0 * y1
    D2 = x3 * y2 - x2 * y3

    y = float(dy1 * D2 - D1 * dy2) / (dy1 * dx2 - dx1 * dy2)
    x = float(y * dx1 - D1) / dy1

    return (int(x), int(y))

def SortPoint(points):
    sp = sorted(points, key = lambda x:(int(x[1]), int(x[0])))
    if sp[0][0] > sp[1][0]:
        sp[0], sp[1] = sp[1], sp[0]
    
    if sp[2][0] > sp[3][0]:
        sp[2], sp[3] = sp[3], sp[2]
    
    return sp

def imgcorr(src):
    rgbsrc = src.copy()
    graysrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurimg = cv2.GaussianBlur(src, (3, 3), 0)
    Cannyimg = cv2.Canny(blurimg, 35, 189)

    lines = cv2.HoughLinesP(Cannyimg, 1, np.pi / 180, threshold = 30, minLineLength = 320, maxLineGap = 40)
   
    for i in range(int(np.size(lines)/4)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(rgbsrc, (x1, y1), (x2, y2), (255, 255, 0), 3)
    
    points = np.zeros((4, 2), dtype = "float32")
    points[0] = CrossPoint(lines[0], lines[2])
    points[1] = CrossPoint(lines[0], lines[3])
    points[2] = CrossPoint(lines[1], lines[2])
    points[3] = CrossPoint(lines[1], lines[3])
    
    sp = SortPoint(points)

    width = int(np.sqrt(((sp[0][0] - sp[1][0]) ** 2) + (sp[0][1] - sp[1][1]) ** 2))
    height = int(np.sqrt(((sp[0][0] - sp[2][0]) ** 2) + (sp[0][1] - sp[2][1]) ** 2))

    dstrect = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]], dtype = "float32")
  
    transform = cv2.getPerspectiveTransform(np.array(sp), dstrect)
    warpedimg = cv2.warpPerspective(src, transform, (width, height))

    return warpedimg

if __name__ == '__main__':
    src = cv2.imread("input.jpg")
    dst = imgcorr(src)
    cv2.imshow("Image", dst)
    cv2.waitKey(0)
    cv2.imwrite("output.jpg", dst)

