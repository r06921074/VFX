
import numpy as np
import cv2
import os, sys, math


def getFocalLength(pano_path):
    focal_length_list = []
    temp = []
    for count, line in enumerate(open(pano_path, 'r', encoding='UTF-8')):
        if count % 13 == 11:
           temp.append(float(line))
    for index, i in enumerate(temp):
        if index == 0:
            focal_length_list.append(temp[-1])
        elif index < len(temp):
            focal_length_list.append(temp[index-1])
    return focal_length_list

def inverseWarping(img, focal_length, feature):
    img_warp = np.zeros(img.shape).astype(np.uint8)
    #focal_length *= 0.7
    s = focal_length
    count_feature_change = 0
    feature_copy = feature.copy()
    for channel in range(img_warp.shape[2]):
        for x_warp in range(img_warp.shape[1]):
            for y_warp in range(img_warp.shape[0]):
                #轉換成以中心當原點
                x_prime = x_warp - (img_warp.shape[1]/2)
                y_prime = y_warp - (img_warp.shape[0]/2)

                x = focal_length * math.tan(x_prime/s)
                y = (y_prime*math.sqrt(x**2+focal_length**2)) / s
                
                #轉回來
                x += (img_warp.shape[1]/2)
                y += (img_warp.shape[0]/2)

                if x < img.shape[1]-1 and x >= 0 and y < img.shape[0]-1 and y >= 0:
                    img_warp[y_warp, x_warp,channel] = bilinearInterpolation(img, y, x, channel)
                    #img_warp[y_warp,x_warp,channel] = img[int(y), int(x), channel]
                    
                    #for feature warping
                    pos = (int(x), int(y))
                    if pos in feature_copy:
                        index = feature.index(pos)
                        #print('change ',feature[index], 'to', (x_warp, y_warp))
                        feature[index] = (x_warp, y_warp)
                        count_feature_change += 1
                        feature_copy[index] = (-1, -1)
    print(count_feature_change)
    
    #return img_warp
    return img_warp, feature

def bilinearInterpolation(img, y, x, channel):
    a = x - math.floor(x)
    b = y - math.floor(y)
    value = (1-a)*(1-b)*img[int(math.floor(y)), int(math.floor(x)), channel] \
            + a*(1-b)*img[int(math.floor(y)), int(math.floor(x))+1, channel] \
            + a*b*img[int(math.floor(y))+1, int(math.floor(x))+1, channel] \
            + (1-a)*b*img[int(math.floor(y))+1, int(math.floor(x)), channel]
    return int(value)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = './images/grail'

    image_warp_dir = os.path.join(image_dir, 'warp/')
    features = np.load(os.path.join(image_dir,'features.npy'))
    
    if not os.path.isdir(image_warp_dir):
        os.mkdir(image_warp_dir)
    focal_length_list = getFocalLength(os.path.join(image_dir,'pano.txt'))

    index = 0
    features_warp = []
    filters = ['pano.txt', 'pano.jpg', 'warp', 'features.npy', 'features_warp.npy', 'descriptors.npy']
    for filename in sorted(os.listdir(image_dir)):
        if filename in filters:
            continue
        
        print(filename)
        #print(features[index])
        image_path = os.path.join(image_dir, filename)
        save_path = os.path.join(image_warp_dir, filename)

        img = cv2.imread(image_path)
        img_warp, feature_warp = inverseWarping(img, focal_length_list[index], features[index])
        features_warp.append(feature_warp)

        cv2.imwrite(save_path, img_warp)
        index += 1
    np.save(os.path.join(image_dir,'features_warp.npy'), features_warp)




