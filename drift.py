import cv2
import numpy as np
import os, sys, math

def inverseWarping(pano, source_img):
    W = pano.shape[1]
    H = pano.shape[0]
    h = source_img.shape[0]
    m = float(H-h) / float(W)
    img_warp = np.zeros((h, W, 3), dtype=np.uint8)

    for channel in range(img_warp.shape[2]):
        for x_warp in range(img_warp.shape[1]):
            for y_warp in range(img_warp.shape[0]):

                x = x_warp
                y = y_warp + m * x_warp
                if x < W-1 and x >= 0 and y < H-1 and y >= 0:

                    img_warp[y_warp, x_warp,channel] = bilinearInterpolation(pano, y, x, channel)

    return img_warp

def bilinearInterpolation(img, y, x, channel):
    a = x - math.floor(x)
    b = y - math.floor(y)
    value = (1-a)*(1-b)*img[int(math.floor(y)), int(math.floor(x)), channel] \
            + a*(1-b)*img[int(math.floor(y)), int(math.floor(x))+1, channel] \
            + a*b*img[int(math.floor(y))+1, int(math.floor(x))+1, channel] \
            + (1-a)*b*img[int(math.floor(y))+1, int(math.floor(x)), channel]
    return int(value)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        img_dir = sys.argv[1]
        pano_path = os.path.join(img_dir, 'panorama.jpg')
        source_img_path = sys.argv[2]
    else:
        img_dir = './images/1_small/warp'
        pano_path = os.path.join(img_dir, 'panorama.jpg')
        source_img_path = './images/1_small/DSC_0033.JPG'

    pano = cv2.imread(pano_path)
    source_img = cv2.imread(source_img_path)

    pano_warp = inverseWarping(pano, source_img)
    pano_drift_path = os.path.join(img_dir, 'panorama_drift.jpg')
    cv2.imwrite(pano_drift_path, pano_warp)