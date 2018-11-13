import os
import cv2
import os.path
import numpy as np

def load_joints(file_name):
    f = open(file_name)
    lines = f.readlines()
    num = int(lines[0])
    raw_joints = lines[1:]

    joints = []
    for i in range(num):
        data = raw_joints[i].split(' ')
        # for j in range(0, 63, 3):
        #     joints.append([float(data[j]), float(data[j+1]), float(data[j+2])])
        tmp = []
        for j in range(63):
            tmp.append(float(data[j]))
        joints.append(tmp)
    # for i in range(10):
    #     print(str(joints[i]))
    return np.array(joints)

def load_images(file_name):
    img = cv2.imread(file_name)
    # cv2.imshow('image', img)
    # print(str(img.shape))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

def load_bin(file_name):
    f = open(file_name, 'rb')
    data = np.fromfile(f, dtype=np.uint32)
    width, height, left, top, right, bottom = data[:6]
    depth = np.zeros((height, width), dtype=np.float32)
    f.seek(4*6)
    data = np.fromfile(f, dtype=np.float32)
    depth[top:bottom, left:right] = np.reshape(data, (bottom-top, right-left))
    f.close()
    header_info = [width, height, left, top, right, bottom]
    return depth

def load_x_data(file_path=''):
    x_data_dict = {}
    img_data_dict = {}
    file_path = '/home/likewise-open/SENSETIME/chengquan/Desktop/occlusion/data/cvpr15_MSRAHandGestureDB/P0/1/'
    for root, dirs, files in os.walk(file_path):
        cnt = 0
        for name in files:
            name_str = name.split('.')
            name_pre = name_str[0]
            name_idx = name_pre.split('_')[0]
            if name_idx != 'joint':
                idx = int(name_idx)
            # jpg or bin
            name_tail = name_str[1]
            if name_tail == 'bin':
                x_data_dict[idx] = load_bin(file_path + name)
            # elif name_tail == 'jpg':
                # img_data_dict[idx] = load_images(file_path + name)

    x_data = []

    for i in range(len(x_data_dict)):
        tmp = x_data_dict[i][np.newaxis, :, :]
        x_data.append(tmp)
    
    return np.array(x_data)
    # return x_data_dict, img_data_dict


def joint_normalization():
    y = load_joints('joint.txt')

    dx = y[:, 0]
    dy = y[:, 1]
    dz = y[:, 2]

    for i in range(3, 63, 3):
        dx = np.append(dx, y[:,i])
        dy = np.append(dy, y[:,i+1])
        dz = np.append(dz, y[:,i+2])

    min_x = dx.min()
    mx = dx.max() - dx.min()
    min_y = dy.min()
    my = dy.max() - dy.min()
    min_z = dz.min()
    mz = dz.max() - dz.min()

    for i in range(y.shape[0]):
        for j in range(0, 63, 3):
            y[i][j] = (y[i][j] - min_x) / mx
            y[i][j+1] = (y[i][j+1] - min_y) / my
            y[i][j+2] = (y[i][j+2] - min_z) / mz
    
    return y



# if __name__ == '__main__':
    # load_joints('joint.txt')
    # load_images('depth.jpg')
    # load_bin('depth.bin')
    # x = load_x_data()
    # print(type(x))
    # # print(x.shape)
    # data = joint_normalization()
    # print(data[1])
    # y = load_joints('joint.txt')[:5]