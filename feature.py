import numpy as np
import matplotlib.pyplot as plt
import cv2
import os,sys

def read_imgs(path='./images/1_small'):
    imgs =[]
    filters = ['pano.txt', 'pano.jpg', 'warp', 'features.npy', 'features_warp.npy', 'descriptors.npy']
    for i in sorted(os.listdir(path)):
        if i in filters:
            continue
        #j = os.path.splitext(i)[0]
        imgs.append(plt.imread(os.path.join(path, i)))
    imgs = np.array(imgs)
    return imgs



def response_R(img,k=0.04):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/255.0
    dy,dx = np.gradient(gray)
    
    Iyy = dy**2
    Ixx = dx**2
    Ixy = dx*dy
    
    Sy2 = cv2.GaussianBlur(Iyy,(3,3),1)
    Sx2 = cv2.GaussianBlur(Ixx,(3,3),1)
    Sxy = cv2.GaussianBlur(Ixy,(3,3),1)
    
    response = np.zeros((img.shape[0],img.shape[1]))

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            M = np.array([[Sx2[y][x], Sxy[y][x]],
                         [Sxy[y][x], Sy2[y][x]]])
            det_M = np.linalg.det(M)
            trace_M = Sx2[y][x] + Sy2[y][x]
            R = det_M - k*(trace_M**2)
            response[y][x] = R
            
    return response



def img_feature(img,response,box_size=5,edge = 20,threshold=0.01):
    response_max = response.max()
    feature_point = np.zeros((img.shape[0],img.shape[1]))
    
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if response[y][x] >= threshold *response_max:
                feature_point[y][x] = response[y][x] 
                
    feature_point[:edge,:] = 0
    feature_point[-edge:,:] = 0
    feature_point[:,-edge:] = 0
    feature_point[:,:edge] = 0
    
    for y in range(0,img.shape[0]-box_size,box_size): 
        for x in range(0,img.shape[1]-box_size,box_size):
            response_radius = feature_point[y:y+box_size,x:x+box_size]
            if (response_radius.sum()) == 0:
                continue
            else:
                max_box_response = np.argmax(response_radius)
                max_response_y,max_response_x = np.unravel_index(max_box_response,response_radius.shape)
                feature_point[y:y+box_size,x:x+box_size] = 0
                feature_point[y+max_response_y,x+max_response_x] = 1  
    return feature_point



def img_descriptor(img,feature):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)/255.0    
    descriptors = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if feature[y][x] == 1:
                patch = img[y-20:y+20,x-20:x+20]
                blur = cv2.GaussianBlur(patch,(3,3),1)
                resize_patch = cv2.resize(blur,(8,8),interpolation=cv2.INTER_AREA)
                normalized_patch = ((resize_patch - resize_patch.mean()) / (resize_patch.std()))
                descriptor = normalized_patch.flatten()
                descriptor = np.append(descriptor,y)
                descriptor = np.append(descriptor,x)
                descriptors.append(descriptor)
    return np.array(descriptors)



def feature_matching(descriptor1,descriptor2,threshold=0.85):
    

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    des1 = np.array(descriptor1[:,:64],dtype=np.float32)
    des2 = np.array(descriptor2[:,:64],dtype=np.float32)
    
    
    matches = bf.knnMatch(des1,des2,k=2)
    
    good_keypoints = []
    
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_keypoints.append(m)

    match_pair = np.zeros((len(good_keypoints),2,2))
    for i in range(len(good_keypoints)):
        idx_2 = good_keypoints[i].trainIdx
        idx_1 = good_keypoints[i].queryIdx
        match_pair[i,0,:] = descriptor1[idx_1,64:66]
        match_pair[i,1,:] = descriptor2[idx_2,64:66]
        
    return match_pair


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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
    else:
        img_dir = './images/1_small'
    imgs = read_imgs(img_dir)
    
    features = []
    descriptors = []
    for i in range(len(imgs)):

        r0 = response_R(imgs[i])
        feature0 = img_feature(imgs[i],r0)
        des0 = img_descriptor(imgs[i],feature0)
        this_feature = []
        feature_pos = np.where(feature0==1)
        for i in range(len(feature_pos[0])):
            this_feature.append((feature_pos[1][i], feature_pos[0][i])) #feature_pos[1][i] = x-axis, feature_pos[0][i] = y-axis

        features.append(this_feature)
        descriptors.append(des0)

    np.save(os.path.join(img_dir, 'features.npy'), features)
    np.save(os.path.join(img_dir, 'descriptors.npy'), descriptors)
