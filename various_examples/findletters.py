import cv2
import numpy as np

# Load the image and the templates
image = cv2.imread('your_image.jpg', 0) # Load in grayscale
E_template = cv2.imread('E_template.jpg', 0)
H_template = cv2.imread('H_template.jpg', 0)

# Store templates in a list
templates = [('E', E_template), ('H', H_template)]

for letter, template in templates:
    w, h = template.shape[::-1]

    # Apply template Matching
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]): # Switch x and y
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)

# Show the final image with the detected letters
cv2.imshow('Detected Letters', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
