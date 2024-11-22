import cv2
import numpy as np
#pip install pillow if you don't have it
import os
import yolov11

#There is a problem with the yellow door showing up as a yellow point, might need to change the yellow color to orange


# Load the image
def patch_overlay(image_path, patch_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for each sphere
    # Magenta color range for top-left corner
    lower_magenta = np.array([140, 50, 50])
    upper_magenta = np.array([160, 255, 255])

    # Yellow color range for top-right corner
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Cyan color range for bottom-right corner
    lower_cyan = np.array([80, 100, 100])
    upper_cyan = np.array([100, 255, 255])

    # Red color range for bottom-left corner (two ranges for red due to HSV wrap-around)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    #Create list for points

    points = []

    # Function to detect and mark points based on color range
    def detect_color_points(hsv_image, lower_range, upper_range, image, color_name):
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            print(f"{color_name} point detected at: ({center_x}, {center_y})")
            points.append((center_x, center_y))
            #cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)

    # Detect each color
    detect_color_points(hsv_image, lower_magenta, upper_magenta, image, "Magenta")
    detect_color_points(hsv_image, lower_yellow, upper_yellow, image, "Yellow")
    detect_color_points(hsv_image, lower_cyan, upper_cyan, image, "Cyan")
    detect_color_points(hsv_image, lower_red1, upper_red1, image, "Red")

    print(points)

    if len(points) == 4:
        #Load overlay image
        overlay_img = cv2.imread(patch_path) # Replace with path to overlay image
            
        #Dimension of overlay image in original format
        src_pts = np.array([[0, 0], [overlay_img.shape[1], 0], [overlay_img.shape[1], overlay_img.shape[0]], [0, overlay_img.shape[0]]], dtype = np.float32)

        #Dimensions of points on base image that we want to transform the overlay image onto

        #Later on, we can adjust the points to be smaller than the size of the entire car roof for a more accurate representation of what we want to do.
        dst_pts = np.array(points, dtype=np.float32)

        #Find the homography matrix between source and destination points
        H, _ = cv2.findHomography(src_pts, dst_pts)

        #Using homography matrix, warps, the overlay image using the H matrix, based on the size of the base image
        warped_img = cv2.warpPerspective(overlay_img, H, (image.shape[1], image.shape[0]))


        #Here, we create a mask to overlay the image, and cut out the size of the overlay. Then we add the overlay on top and show it.
        gray_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)
        base_img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
        final_img = cv2.add(base_img_bg, warped_img)

        print(H)

        #Write homography matrix and pixels to a csv or other type of file

        cv2.imshow('Final Image', final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Inconsistent points for {image_path}")

    # Display the result
    #cv2.imshow('Detected Points', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def process_images(img_folder, patch_path):
    for filename in os.listdir(img_folder):
        if filename.endswith('.png'):
            file_path = os.path.join(img_folder, filename)
            patch_overlay(file_path, patch_path)

# Testing

# Replace 'images' with your images folder, and 'patch.jpg' with the patch image
process_images('images', 'patch.jpg')

# patch_overlay('image_0.png', 'patch.jpg')
# patch_overlay('image_1.png', 'patch.jpg')
# patch_overlay('image_2.png', 'patch.jpg')