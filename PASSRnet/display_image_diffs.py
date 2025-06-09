import cv2
import numpy as np

def highlight_differences(image_path1, image_path2):
    # Load images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded.")
        return

    # Resize images to match (if needed)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Compute absolute difference for each channel
    diff = cv2.absdiff(img1, img2)

    # Convert difference image to grayscale
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold the difference image to get a binary mask
    DIFFERENCE_THRESHOLD = 40
    _, mask = cv2.threshold(gray_diff, DIFFERENCE_THRESHOLD, 255, cv2.THRESH_BINARY)

    num_different_pixels = np.count_nonzero(mask)
    print(f"Number of different pixels: {num_different_pixels}")

    # Create a red overlay where differences exist
    highlighted = img1.copy()
    highlighted[mask > DIFFERENCE_THRESHOLD] = [0, 0, 255]  # Set different pixels to red

    cv2.imwrite(f"difference_pixels/{model_name}_{DIFFERENCE_THRESHOLD}.png", highlighted)
    # Show results
    cv2.imshow('Differences Highlighted', highlighted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
#model_name = "PASSR_MMHCA2"
#model_name = "PASSR_normal2"
model_name = "INTRE_ELE2"
original_image_path = "/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/test/Middlebury/hr/pipes/hr0.png"
image_path1 = "/home/david/PycharmProjects/SSRDEF-Net/PASSRnet/results/PASSR_MHCA4_4xSR_iter61501/Middlebury/pipes_L.png"
image_path2 = "/home/david/PycharmProjects/SSRDEF-Net/PASSRnet/results/PASSRnet_x4/Middlebury/pipes_L.png"
highlight_differences(original_image_path, image_path1)
