import numpy as np
import cv2
import os, sys, math, random
import matplotlib.pyplot as plt


def matched_pairs_plot(p1, p2, mp):
    _, offset, _ = p1.shape
    plt_img = np.concatenate((p1, p2), axis=1)
    plt.figure(figsize=(10,10))
    plt.imshow(plt_img)
    for i in range(len(mp)):
        plt.scatter(x=mp[i][0][1], y=mp[i][0][0], c='r')
        plt.plot([mp[i][0][1], offset+mp[i][1][1]], [mp[i][0][0], mp[i][1][0]], 'y-', lw=1)
        plt.scatter(x=offset+mp[i][1][1], y=mp[i][1][0], c='b')
    plt.show()

def featureMatching(descriptors):

    feature_pairs = []
    for i in range(len(descriptors)-1):
        feature_pair = []
        min_dist_list = []
        for index1, descriptor1 in enumerate(descriptors[i]):
            min_dist = 999999
            for index2, descriptor2 in enumerate(descriptors[i+1]):
                dist = np.linalg.norm(descriptor1[:64] - descriptor2[:64])
                
                if dist < min_dist:
                    min_dist = dist
                    min_index2 = index2
            min_dist_list.append(min_dist)
            feature_pair.append(min_index2)

        min_dist = min(min_dist_list)
        for index, dist in enumerate(min_dist_list):
            if dist > min_dist * 10:
                feature_pair[index] = -1


        
        feature_pairs.append(feature_pair)
    return feature_pairs
    #feature_pairs[0][5] = k, 代表第0張圖的第5個feature對應到第1張圖的第k個feature

def ransac(features, feature_pairs, img_index):
    k=200
    threshold = 5
    best_x_move = 0
    best_y_move = 0
    best_count_inliner = 0
    pairs = list(enumerate(feature_pairs[img_index]))
    good_pairs = [x for x in pairs if x[1] != -1]
    for i in range(k):
        
        sample = random.choice(good_pairs)

        this_index = sample[0]
        this_feature = features[img_index][this_index]
        paired_index = sample[1]
        paired_feature = features[img_index+1][paired_index]
        x_move = paired_feature[0] - this_feature[0]
        y_move = paired_feature[1] - this_feature[1]
        if abs(y_move) > 50:
            continue
        
        count_inliner = 0
        for j, this_feature in enumerate(features[img_index]):
            paired_index = feature_pairs[img_index][j]
            if paired_index == -1:
                continue
            paired_feature = features[img_index+1][paired_index]
            diff = math.sqrt( (paired_feature[0]-(this_feature[0]+x_move))**2 + (paired_feature[1]-(this_feature[1]+y_move))**2 )
            if diff < threshold:
                count_inliner += 1
            #print(diff)
        if count_inliner > best_count_inliner:
            best_x_move = x_move
            best_y_move = y_move
            best_count_inliner = count_inliner
    print('best_move = ',best_x_move, best_y_move)
    print('best_count_inliner = ',best_count_inliner)
    return best_x_move, best_y_move

#將img1接到img2右方
def imageStitching(img1, img2, best_x_move, best_y_move, accumulate_move_y):
    global max_accumulate_move_y, min_accumulate_move_y
    if accumulate_move_y > max_accumulate_move_y:
        new_height = img1.shape[0] + (accumulate_move_y - max_accumulate_move_y)
        height_add = accumulate_move_y - max_accumulate_move_y
    elif accumulate_move_y < min_accumulate_move_y:
        new_height = img1.shape[0] + abs(accumulate_move_y - min_accumulate_move_y)
        height_add = abs(accumulate_move_y - min_accumulate_move_y)
    else:
        new_height = img1.shape[0]
        height_add = 0
    print('###############################')
    print('best_y_move=', best_y_move)
    print('height_add=',height_add)
    print('new_height=',new_height)
    print('accumulate_move_y=', accumulate_move_y)
    if accumulate_move_y > max_accumulate_move_y:
        max_accumulate_move_y = accumulate_move_y
    elif accumulate_move_y < min_accumulate_move_y:
        min_accumulate_move_y = accumulate_move_y
    print('max_accumulate_move_y=', max_accumulate_move_y)
    print('min_accumulate_move_y=', min_accumulate_move_y)
    new_img = np.zeros((new_height, img1.shape[1]+best_x_move, 3), dtype=np.uint8)
    for channel in range(new_img.shape[2]):
        for x in range(new_img.shape[1]):
            for y in range(new_img.shape[0]):
                if best_y_move >= 0:
                        #img2區域
                    if (x < best_x_move and y >= abs(accumulate_move_y) and y < (img2.shape[0])) or (x >= best_x_move and x < img2.shape[1] and y > abs(accumulate_move_y) and y < new_height-img1.shape[0]):
                        new_img[y, x, channel] = img2[y-abs(accumulate_move_y), x, channel]
                    
                        #重疊區域
                    elif x >= best_x_move and x < img2.shape[1] and y > abs(accumulate_move_y) and y < img2.shape[0]:
                        #Linear Blending 
                        
                        w1 = (x - best_x_move) / (img2.shape[1]-best_x_move) #越靠近img1, w1就越大
                        w2 = 1 - ((x - best_x_move) / (img2.shape[1]-best_x_move)) #越靠近img1, w2就越小
                        new_img[y, x, channel] = w1 * img1[y-height_add, x-best_x_move, channel] \
                                                + w2 * img2[y-abs(accumulate_move_y), x, channel]           
                        
                            #直接用img1
                        #new_img[y, x, channel] = img1[y-height_add, x-best_x_move, channel]
                            #直接用img2
                        #new_img[y, x, channel] = img2[y-abs(accumulate_move_y), x, channel]

                        #img1區域
                    elif (x >= (img2.shape[1]) and y >=height_add) or (x >= best_x_move and x < img2.shape[1] and y >= img2.shape[0]):
                        new_img[y, x, channel] = img1[y-height_add, x-best_x_move, channel]
                #best_y_move < 0        
                else:
                        #img2區域
                    if (x < best_x_move and y > new_img.shape[0]-img2.shape[0]) or (x >= best_x_move and x < img2.shape[1] and y >= img1.shape[0]):
                        new_img[y, x, channel] = img2[y-(new_img.shape[0]-img2.shape[0]), x, channel]
                    
                        #重疊區域
                    elif x >= best_x_move and x < img2.shape[1] and y >= new_img.shape[0]-img2.shape[0] and y < img1.shape[0]:    
                    #elif x >= best_x_move and x < img2.shape[1] and y > new_img.shape[0]-img2.shape[0] and y < (new_img.shape[0]+best_y_move):
                        #Linear Blending 
                        
                        w1 = (x - best_x_move) / (img2.shape[1]-best_x_move) #越靠近img1, w1就越大
                        w2 = 1 - ((x - best_x_move) / (img2.shape[1]-best_x_move)) #越靠近img1, w2就越小
                        new_img[y, x, channel] = w1 * img1[y, x-best_x_move, channel] \
                                                + w2 * img2[y-(new_img.shape[0]-img2.shape[0]), x, channel]      
                                   
                            #直接用img1
                        #new_img[y, x, channel] = img1[y, x-best_x_move, channel]
                            #直接用img2
                        #new_img[y, x, channel] = img2[y-(new_img.shape[0]-img2.shape[0]), x, channel]
                        #img1區域
                    elif (x >= (img2.shape[1]) and y < new_img.shape[0]-height_add) or (x >= best_x_move and x < img2.shape[1] and y < new_img.shape[0]-img2.shape[0]):
                        new_img[y, x, channel] = img1[y, x-best_x_move, channel]
    return new_img



if __name__ == '__main__':

    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
        img_warp_dir = os.path.join(sys.argv[1], 'warp/')
    else:
        img_dir = './images/grail/'
        img_warp_dir = './images/grail/warp/'

    img_list = []
    for filename in sorted(os.listdir(img_warp_dir)):
        if filename == 'panorama.jpg' or filename == 'panorama_drift.jpg' or filename == 'panorama_refine.jpg' or filename == 'panorama_clip.jpg':
            continue
        file_path = img_warp_dir + filename
        img_list.append(cv2.imread(file_path))
    
    
    features = np.load(os.path.join(img_dir, 'features.npy'))
    descriptors = np.load(os.path.join(img_dir, 'descriptors.npy'))
    feature_pairs = featureMatching(descriptors)
    
    best_x_moves = []
    best_y_moves = []
    
    for i in range(len(img_list)-1):
        best_x_move, best_y_move = ransac(features, feature_pairs, i)
        best_x_moves.append(best_x_move)
        best_y_moves.append(best_y_move)
    mean_error_y = int(sum(best_y_moves)/len(img_list))
    print('mean_error_y = ', mean_error_y)
    
    np.save('best_x_moves1.npy', best_x_moves)
    np.save('best_y_moves1.npy', best_y_moves)

    new_img = img_list[0]
    accumulate_move_y = 0
    max_accumulate_move_y = 0
    min_accumulate_move_y = 0

    for i in range(len(img_list)-1):
        
        #減掉平均move_y
        #best_y_moves[i] -= mean_error_y
        accumulate_move_y += best_y_moves[i]

        new_img = imageStitching(new_img, img_list[i+1], best_x_moves[i], best_y_moves[i], accumulate_move_y)

    cv2.imshow('a',new_img)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(img_warp_dir, 'panorama.jpg'), new_img)

