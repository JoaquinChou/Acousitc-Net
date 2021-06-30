import cv2


# Convert an image with a resolution of 1024×513 to 1024×512
def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (1024, 512))

    return img
