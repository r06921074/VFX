import cv2, os, sys
import numpy as np


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = 'images/1_small/warp'
    pano = cv2.imread(os.path.join(image_dir, 'panorama.jpg'))
    print(pano.shape)

    if len(sys.argv) > 3:
        top_clip_ratio = sys.argv[2]
        bottom_clip_ratio = sys.argv[3]
    else:
        top_clip_ratio = 0.15
        bottom_clip_ratio = 0.7


    h = pano.shape[0]
    pano = np.delete(pano, slice(0, int(h*top_clip_ratio)), axis=0)
    pano = np.delete(pano, slice(int(h*bottom_clip_ratio), h), axis=0)
    cv2.imwrite(os.path.join(image_dir, 'panorama_clip.jpg'), pano)