import cv2
import numpy as np

def histogram_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def unsharp_mask(img, sigma=1.0, amount=1.0):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return sharpened

def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def on_trackbar_change(val):
    global enhanced_image
    global final_image

    sigma = cv2.getTrackbarPos('Sigma', 'Enhanced Image') / 10
    amount = cv2.getTrackbarPos('Amount', 'Enhanced Image') / 10
    gamma = cv2.getTrackbarPos('Gamma', 'Enhanced Image') / 10

    final_image = unsharp_mask(enhanced_image, sigma, amount)
    final_image = gamma_correction(final_image, gamma)

    cv2.imshow('Enhanced Image', final_image)

# Load the image
input_image = cv2.imread('2.2.07.tiff')

# Enhance the image
enhanced_image = histogram_equalization(input_image)

# Create a window to display the enhanced image
cv2.namedWindow('Enhanced Image')

# Create trackbars for sigma, amount, and gamma
cv2.createTrackbar('Sigma', 'Enhanced Image', 40, 100, on_trackbar_change)
cv2.createTrackbar('Amount', 'Enhanced Image', 2, 100, on_trackbar_change)
cv2.createTrackbar('Gamma', 'Enhanced Image', 6, 20, on_trackbar_change)

# Display the enhanced image with initial unsharp masking and gamma correction parameters
final_image = unsharp_mask(enhanced_image, 4, 0.2)
final_image = gamma_correction(final_image, 0.6)

cv2.imshow('Enhanced Image', final_image)
cv2.imshow('Ori Image', input_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final image with the selected unsharp masking and gamma correction parameters
cv2.imwrite('oakland_harbor_enhanced.tiff', final_image)
