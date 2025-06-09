import cv2
import os

def upscale_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image could not be loaded.")
        return

    # Get original dimensions
    height, width = img.shape[:2]
    SCALE = 4

    # Downscale and then upscale using bicubic interpolation
    downscaled = cv2.resize(img, (width // SCALE, height // SCALE), interpolation=cv2.INTER_CUBIC)
    upscaled = cv2.resize(downscaled, (width, height), interpolation=cv2.INTER_CUBIC)

    # Get image name from path
    image_name = os.path.basename(image_path)

    # Create output folder if it doesn't exist
    os.makedirs('bicubic_saved', exist_ok=True)

    # Save the image
    cv2.imwrite(f'bicubic_saved/{image_name}', upscaled)
    print(f"Saved as bicubic_saved/{image_name}")

# Example usage
image_path = "/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/test/Middlebury/hr/pipes/hr0.png"
upscale_image(image_path)
