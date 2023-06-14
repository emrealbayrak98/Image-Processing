import cv2
import numpy as np

def guided_filter(I, p, r, eps):
    I_mean = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    p_mean = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    Ip_mean = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    I_variance = cv2.boxFilter(I * I, cv2.CV_64F, (r, r)) - I_mean * I_mean
    Ip_covariance = Ip_mean - I_mean * p_mean
    a = Ip_covariance / (I_variance + eps)
    b = p_mean - a * I_mean
    a_mean = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    b_mean = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = a_mean * I + b_mean
    return q

def histogram_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def high_boost_filter(img, sigma=1.0, amount=1.0):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)

def gamma_correction(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def enhance_image(image_path, omega, t0, w_size, r, eps, gamma, sigma, amount):
    # Load the image
    img = cv2.imread(image_path)

    # Apply histogram equalization using CLAHE
    img_equalized = histogram_equalization(img)

    # Apply high-boost filter to sharpen the image
    img_sharpened = high_boost_filter(img_equalized, sigma, amount)
    
    img_gamma_corrected = gamma_correction(img_sharpened, gamma)

    # Dehazing the image using Dark Channel Prior method
    def get_dark_channel(img, w_size):
        min_channel = np.amin(img, axis=2)
        dark_channel = cv2.erode(min_channel, np.ones((w_size, w_size)))
        return dark_channel

    def estimate_atmospheric_light(img, dark_channel, percent=0.01):
        num_pixels = dark_channel.size
        num_brightest = int(num_pixels * percent)
        indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
        brightest_pixels = img.reshape((-1, 3))[indices]
        return np.max(brightest_pixels, axis=0)

    def dehaze(img, omega, t0, w_size, r, eps):
        dark_channel = get_dark_channel(img, w_size)
        A = estimate_atmospheric_light(img, dark_channel)
        t = 1 - omega * (dark_channel[:, :, np.newaxis] / A)
        t = np.maximum(t, t0)
        
        # Refine the transmission map using a simpler approach
        t_refined = cv2.bilateralFilter(t, 9, 75, 75)
        
        J = np.zeros_like(img, dtype=np.float64)
        for i in range(3):
            J[:,:,i] = ((img[:,:,i] - A[i]) / t_refined[:,:,i]) + A[i]
        J = np.clip(J, 0, 255).astype(np.uint8)
        return J

    img_dehazed = dehaze(img_gamma_corrected, omega, t0, w_size, r, eps)
    
    return img_dehazed

# Load your foggy forest road photo
image_path = 'haze.png'

# Read the image
img = cv2.imread(image_path)

# Create windows for the original and enhanced images
cv2.namedWindow('Original Image')
cv2.namedWindow('Enhanced Image')

# Display the original and enhanced images
cv2.imshow('Original Image', img)

enhanced_image = enhance_image(image_path, omega=0.9, t0=0.1, w_size=15, r=50, eps=1e-3, gamma=1.0, sigma=1.0, amount=1.0)
cv2.imshow('Enhanced Image', enhanced_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()