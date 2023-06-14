import cv2
import numpy as np

image_path = 'moliets.png'
# Load the image
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define default color range and area thresholds
lower_blue = np.array([66, 37, 76])
upper_blue = np.array([126, 218, 255])
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 9, 255])
min_pool_area = 113
max_pool_area = 500

# Threshold the HSV image to get blue and white colors
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

# Combine the masks
combined_mask = cv2.bitwise_or(blue_mask, white_mask)

# Apply morphological operations to remove noise and small objects
kernel = np.ones((3, 3), np.uint8)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the binary image
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out contours based on area thresholds
filtered_contours = [c for c in contours if min_pool_area < cv2.contourArea(c) < max_pool_area]

# Draw the detected pools on the original image
result_image = image.copy()
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draw the number of detected pools on the image
text = f'Number of pools detected: {len(filtered_contours)}'
text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
text_x = result_image.shape[1] - text_size[0] - 10
text_y = result_image.shape[0] - text_size[1] - 10
cv2.putText(result_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show the final result
cv2.imshow("Detected Pools", result_image)
cv2.imwrite("pooldetection.png", result_image)

# Wait for any key to be pressed and then close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()