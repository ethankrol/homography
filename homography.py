import cv2
import numpy as np

#Function creating points based on mouse click. Right now, points must be defined in a clockwise or counterclockwise order.
def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONUP and len(points) < 4:
        points.append([x, y])

#Reading in base image
base_img = cv2.imread('frame_at_second_00072.jpg') # Replace with path to base image
cv2.namedWindow('Base Image')

#Initialize points list to empty list, and exit when points list has 4 points
points = []
cv2.setMouseCallback('Base Image', mouse_callback)

while True:
    for point in points:
        cv2.circle(base_img, tuple(point), 5, (0, 0, 255), -1) #Draws circle with radius 5, color (0, 0, 255). -1 fills in circle
    cv2.imshow('Base Image', base_img)
    if len(points) == 4:
        break
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

#Load overlay image
overlay_img = cv2.imread('patch.jpg') # Replace with path to overlay image
    
#Dimension of overlay image in original format
src_pts = np.array([[0, 0], [overlay_img.shape[1], 0], [overlay_img.shape[1], overlay_img.shape[0]], [0, overlay_img.shape[0]]], dtype = np.float32)

#Dimensions of points on base image that we want to transform the overlay image onto

#Later on, we can adjust the points to be smaller than the size of the entire car roof for a more accurate representation of what we want to do.
dst_pts = np.array(points, dtype=np.float32)

#Find the homography matrix between source and destination points
H, _ = cv2.findHomography(src_pts, dst_pts)

#Using homography matrix, warps, the overlay image using the H matrix, based on the size of the base image
warped_img = cv2.warpPerspective(overlay_img, H, (base_img.shape[1], base_img.shape[0]))


#Here, we create a mask to overlay the image, and cut out the size of the overlay. Then we add the overlay on top and show it.
gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)

mask_inv = cv2.bitwise_not(mask)
base_img_bg = cv2.bitwise_and(base_img, base_img, mask=mask_inv)
final_img = cv2.add(base_img_bg, warped_img)

cv2.imshow('Final Image', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()