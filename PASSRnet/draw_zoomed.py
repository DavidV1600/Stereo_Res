import matplotlib.pyplot as plt
import cv2

# Load your image
#MODEL_NAME = "BICUBIC"
#MODEL_NAME = "My-PASSR"
MODEL_NAME = "PASSR"
#IMG_PATH = "/home/david/PycharmProjects/SSRDEF-Net/PASSRnet/bicubic_saved/hr0.png"
#IMG_PATH = "/home/david/PycharmProjects/SSRDEF-Net/PASSRnet/results/PASSR_MHCA4_4xSR_iter61501/Middlebury/pipes_L.png"
IMG_PATH = "/home/david/PycharmProjects/SSRDEF-Net/PASSRnet/results/PASSRnet_x4/Middlebury/pipes_L.png"
#IMG_PATH = "/home/david/PycharmProjects/SSRDEF-Net/SSRDEFNet-PyTorch/data/test/Middlebury/hr/pipes/hr0.png"
img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define region of interest (x, y, width, height)
x, y, w, h = 350, 350, 50, 50
zoomed = img[y:y+h, x:x+w]

# Resize zoomed region
zoom_factor = 3
zoomed_resized = cv2.resize(zoomed, (w*zoom_factor, h*zoom_factor), interpolation=cv2.INTER_NEAREST)

# Plotting
fig, ax = plt.subplots()
ax.imshow(img)
# Rectangle around ROI
rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
ax.add_patch(rect)

# Add inset
inset_ax = fig.add_axes([0.6, 0.6, 0.25, 0.25])  # [left, bottom, width, height]
inset_ax.imshow(zoomed_resized)
inset_ax.set_xticks([])
inset_ax.set_yticks([])

plt.savefig(f'zoomed_images/{MODEL_NAME}', bbox_inches='tight', dpi=300)
