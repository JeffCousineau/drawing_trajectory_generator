# Reference
# draw in javascript : http://www.williammalone.com/articles/create-html5-canvas-javascript-drawing-app/#demo-simple
# zhang-suen algorithm : https://github.com/bsdnoobz/zhang-suen-thinning/blob/master/thinning.py
# houghlines : https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghlines

import weave
import numpy as np
import cv2

def _thinningIteration(im, iter):
	I, M = im, np.zeros(im.shape, np.uint8)
	expr = """
	for (int i = 1; i < NI[0]-1; i++) {
		for (int j = 1; j < NI[1]-1; j++) {
			int p2 = I2(i-1, j);
			int p3 = I2(i-1, j+1);
			int p4 = I2(i, j+1);
			int p5 = I2(i+1, j+1);
			int p6 = I2(i+1, j);
			int p7 = I2(i+1, j-1);
			int p8 = I2(i, j-1);
			int p9 = I2(i-1, j-1);
			int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
			         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
			         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
			         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
			if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
				M2(i,j) = 1;
			}
		}
	} 
	"""

	weave.inline(expr, ["I", "iter", "M"])
	return (I & ~M)


def thinning(src):
	dst = src.copy() / 255
	prev = np.zeros(src.shape[:2], np.uint8)
	diff = None

	while True:
		dst = _thinningIteration(dst, 0)
		dst = _thinningIteration(dst, 1)
		diff = np.absolute(dst - prev)
		prev = dst.copy()
		if np.sum(diff) == 0:
			break

	return dst * 255

if __name__ == "__main__":

	# OPEN IMAGE

	src = cv2.imread("sara.jpeg")
	bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	_, bw2 = cv2.threshold(bw, 10, 255, cv2.THRESH_BINARY)

	# INVERT FOR THINNING
	bw2 = cv2.bitwise_not(bw2)

	# ZHUANG THINNING ALGORITHM
	bw2 = thinning(bw2)

	# REINVERT
	#bw2 = cv2.bitwise_not(bw2)

	# FIND LINES
	#edges = cv2.Canny(bw2, 50, 150, apertureSize=3)
	threshold = 10
	minLineLength = 5
	lines = cv2.HoughLinesP(bw2, 1, np.pi / 180, threshold, 0, minLineLength, 20);

	# Draw lines by starting by the first one
	# Then find the closest point to draw next
	already_draw = []

	for line in lines:
		already_draw.append(line)
		print(line)
		cv2.line(src, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2)
		cv2.imshow("draw line result", src)
		cv2.waitKey()



	cv2.imshow("thining result", bw2)
	cv2.imshow("draw line result", src)
	print("NUMBER OF LINES :", len(lines))
	cv2.waitKey()
