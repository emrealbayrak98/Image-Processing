import cv2
import matplotlib.pyplot as plt
# Load an example image
image_url = 'picture.jpeg'
image = cv2.imread(image_url)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Perform thresholding with values 0.25, 0.5, and 0.75
threshold_values = [int(255 * t) for t in [0.25, 0.5, 0.75]]
_, thresholded_images = zip(*[cv2.threshold(gray_image, threshold, 255,
cv2.THRESH_BINARY) for threshold in threshold_values])
# Perform adaptive thresholding with block sizes 3 and 15
block_sizes = [3, 15]
adaptive_thresholded_images = [cv2.adaptiveThreshold(gray_image, 255,
cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, -5) for blockSize
in block_sizes]
# Plot the original and fixed thresholded images
plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
for i, thresholded_image in enumerate(thresholded_images, 2):
    plt.subplot(1, 4, i)
    plt.imshow(thresholded_image, cmap='gray')
    plt.title(f'Threshold: {threshold_values[i - 2] / 255:.2f}')
plt.show()
# Plot the adaptive thresholded images
plt.figure(figsize=(8, 4))
for i, adaptive_thresholded_image in enumerate(adaptive_thresholded_images,
1):
    plt.subplot(1, 2, i)
    plt.imshow(adaptive_thresholded_image, cmap='gray')
    plt.title(f'Adaptive Threshold: {block_sizes[i - 1]} Block Size')
plt.show()