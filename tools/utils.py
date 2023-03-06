import numpy  as np
import random

def random_sample(sample_num, pc_inbox, pc_outbox):
    pc_inbox_label = np.concatenate((pc_inbox, np.ones((1, pc_inbox.shape[1]))), axis=0)
    pc_outbox_label = np.concatenate((pc_outbox, np.zeros((1, pc_outbox.shape[1]))), axis=0)
    pc_with_label = np.concatenate((pc_inbox_label, pc_outbox_label), axis=1)

    a = np.arange(pc_with_label.shape[1])
    mask = random.sample(list(a), sample_num)
    point_coords = pc_with_label[:2, mask].T.tolist()
    point_label = pc_with_label[2, mask].T.tolist()
    return point_coords, point_label

def random_sample_balance(sample_num, pc_inbox, pc_outbox, bbox):
    xc, yc, w, h = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]
    pc_inbox, pc_outbox = pc_inbox.T, pc_outbox.T  # [N, 2]

    # padding or down-sampling to same size
    if pc_inbox.shape[0] >= sample_num:
        idx = np.random.choice(pc_inbox.shape[0], sample_num, replace=False)
    elif pc_inbox.shape[0] == 0:
        pc_inbox = (np.random.rand(sample_num, 2) - 0.5) * min(w, h) / 2 + np.array([xc, yc])
        idx = np.random.choice(pc_inbox.shape[0], sample_num, replace=False)
    else:
        idx = np.random.choice(pc_inbox.shape[0], sample_num, replace=True)
    pc_inbox = pc_inbox[idx, :]

    if pc_outbox.shape[0] >= sample_num:
        idx = np.random.choice(pc_outbox.shape[0], sample_num, replace=False)
    elif pc_outbox.shape[0] == 0:
        pc_outbox = np.random.rand(sample_num, 2) * min(w, h) / 5 + np.array([bbox[0], bbox[1]])
        idx = np.random.choice(pc_outbox.shape[0], sample_num, replace=False)
    else:
        idx = np.random.choice(pc_outbox.shape[0], sample_num, replace=True)
    pc_outbox = pc_outbox[idx, :]

    pc_inbox_label = np.concatenate((pc_inbox, np.ones((pc_inbox.shape[0], 1))), axis=1)
    pc_outbox_label = np.concatenate((pc_outbox, np.zeros((pc_outbox.shape[0], 1))), axis=1)

    pc_with_label = np.concatenate((pc_inbox_label, pc_outbox_label), axis=0)
    np.random.shuffle(pc_with_label)
    point_coords = pc_with_label[:, :2].tolist()
    point_label = pc_with_label[:, 2].tolist()
    return point_coords, point_label

def random_sample_balance2(sample_num, pc_inbox, pc_outbox, bbox):
    xc, yc, w, h = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]
    pc_inbox, pc_outbox = pc_inbox.T, pc_outbox.T  # [N, 2]

    # padding or down-sampling to same size
    if pc_inbox.shape[0] >= sample_num:
        idx = np.random.choice(pc_inbox.shape[0], sample_num, replace=False)
        pc_inbox = pc_inbox[idx, :]
        pc_inbox_label = np.ones((sample_num, 1)) 
    elif pc_inbox.shape[0] == 0:
        pc_inbox = (np.random.rand(sample_num, 2) - 0.5) * min(w, h) / 2 + np.array([xc, yc])
        pc_inbox_label = -np.ones((sample_num, 1)) 
    else:
        idx = np.random.choice(pc_inbox.shape[0], sample_num, replace=True)
        pc_inbox = pc_inbox[idx, :]
        pc_inbox_label = np.ones((sample_num, 1)) 


    if pc_outbox.shape[0] >= sample_num:
        idx = np.random.choice(pc_outbox.shape[0], sample_num, replace=False)
        pc_outbox = pc_outbox[idx, :]
        pc_outbox_label = np.zeros((sample_num, 1)) 
    elif pc_outbox.shape[0] == 0:
        pc_outbox = np.random.rand(sample_num, 2) * min(w, h) / 5 + np.array([bbox[0], bbox[1]])
        pc_outbox_label = -np.ones((sample_num, 1)) 
    else:
        idx = np.random.choice(pc_outbox.shape[0], sample_num, replace=True)
        pc_outbox = pc_outbox[idx, :]
        pc_outbox_label = np.zeros((sample_num, 1)) 

    pc_inbox_with_label = np.concatenate((pc_inbox, pc_inbox_label), axis=1)
    pc_outbox_with_label = np.concatenate((pc_outbox, pc_outbox_label), axis=1)

    pc_with_label = np.concatenate((pc_inbox_with_label, pc_outbox_with_label), axis=0)
    np.random.shuffle(pc_with_label)
    point_coords = pc_with_label[:, :2].tolist()
    point_label = pc_with_label[:, 2].tolist()
    return point_coords, point_label

def random_sample_balance3(sample_num, pc_inbox, pc_outbox, bbox):
    xc, yc, w, h = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]
    pc_inbox, pc_outbox = pc_inbox.T, pc_outbox.T  # [N, 2]

    # padding or down-sampling to same size
    if pc_inbox.shape[0] >= sample_num:
        idx = np.random.choice(pc_inbox.shape[0], sample_num, replace=False)
        pc_inbox = pc_inbox[idx, :]
        pc_inbox_label = np.ones((sample_num, 1))
    elif pc_inbox.shape[0] == 0:
        pc_inbox = (np.random.rand(sample_num, 2) - 0.5) * min(w, h) / 2 + np.array([xc, yc])
        pc_inbox_label = np.ones((sample_num, 1))
    else:
        idx = np.random.choice(pc_inbox.shape[0], sample_num, replace=True)
        pc_inbox = pc_inbox[idx, :]
        pc_inbox_label = np.ones((sample_num, 1))

    if pc_outbox.shape[0] >= sample_num:
        idx = np.random.choice(pc_outbox.shape[0], sample_num, replace=False)
        pc_outbox = pc_outbox[idx, :]
        pc_outbox_label = np.zeros((sample_num, 1))
    elif pc_outbox.shape[0] == 0:
        pc_outbox = np.random.rand(sample_num, 2) * min(w, h) / 5 + np.array([bbox[0], bbox[1]])
        pc_outbox_label = -np.ones((sample_num, 1))
    else:
        idx = np.random.choice(pc_outbox.shape[0], sample_num, replace=True)
        pc_outbox = pc_outbox[idx, :]
        pc_outbox_label = np.zeros((sample_num, 1))

    pc_inbox_with_label = np.concatenate((pc_inbox, pc_inbox_label), axis=1)
    pc_outbox_with_label = np.concatenate((pc_outbox, pc_outbox_label), axis=1)

    pc_with_label = np.concatenate((pc_inbox_with_label, pc_outbox_with_label), axis=0)
    np.random.shuffle(pc_with_label)
    point_coords = pc_with_label[:, :2].tolist()
    point_label = pc_with_label[:, 2].tolist()
    return point_coords, point_label

def remove_overlap(depth_img):
    """
    # remove the overlap points in the projected image
    """
    k_size = 30  # 30
    hor_step = 10  # 10
    ver_step = 5
    height, width = depth_img.shape
    thresh_ratio = 0.25  # 0.25 0.1-0.5

    for i in range(200, height, ver_step):
        for j in range(0, width, hor_step):
            if i >= height - k_size - 1 or j >= width - k_size - 1:
                continue
            value = depth_img[i:i + k_size, j:j + k_size]
            point_idx = np.where(value != 0)

            if len(point_idx[0]) <= 1:
                continue
            else:
                point_loc_list, depth_list = [], []
                point_i = point_idx[0] + i
                point_j = point_idx[1] + j
                for k in range(len(point_j)):
                    depth_list.append(depth_img[point_i[k], point_j[k]])
                    point_loc_list.append((point_i[k], point_j[k]))
                depth_list = np.array(depth_list)
                point_loc_list = np.array(point_loc_list)
                depth_min = depth_list.min()
                depth_max = depth_list.max()
                if (depth_max - depth_min) / depth_min < thresh_ratio:
                    continue
                min_idx = np.where(value == depth_min)
                if min_idx[0][0] == k_size - 1:
                    continue

                depth_minus = (depth_list - depth_min) / depth_min
                idx_near = np.where(depth_minus < thresh_ratio)[0]
                idx_far = np.where(depth_minus >= thresh_ratio)[0]

                if len(idx_near) > 1:
                    pix_list = point_loc_list[idx_near]
                    hor_min = pix_list[:, 1].min() - k_size / 2
                    hor_max = pix_list[:, 1].max() + k_size / 2
                    ver_min = pix_list[:, 0].min() - 1
                else:
                    pix_list = point_loc_list[idx_near]
                    hor_min = pix_list[:, 1][0] - k_size / 2
                    hor_max = hor_min + k_size
                    ver_min = pix_list[:, 0][0] - 1

                for p in range(len(idx_far)):
                    if point_loc_list[idx_far[p]][1] >= hor_min and \
                            point_loc_list[idx_far[p]][1] <= hor_max:
                        # point_loc_list[idx_far[p]][0] >= ver_min:
                        depth_img[point_loc_list[idx_far[p]][0], point_loc_list[idx_far[p]][1]] = 0.

    return depth_img


def keep_max_random(pc_concate, max_num=100):
    pc_concate = pc_concate.T  # [N, 2]

    if pc_concate.shape[0] > max_num:
        idx = np.random.choice(pc_concate.shape[0], max_num, replace=False)
        pc_concate = pc_concate[idx, :]
    return pc_concate


def keep_max(pc_concate, max_num=100):
    pc_concate = pc_concate.T  # [N, 2]

    if pc_concate.shape[0] > max_num:
        pc_concate = pc_concate[:100, :]
    return pc_concate

def keep_all(pc_inbox, pc_outbox):
    pc_inbox, pc_outbox = pc_inbox.T, pc_outbox.T  # [N, 2]
    pc_inbox_label = np.concatenate((pc_inbox, np.ones((pc_inbox.shape[0], 1))), axis=1)
    pc_outbox_label = np.concatenate((pc_outbox, np.zeros((pc_outbox.shape[0], 1))), axis=1)
    pc_with_label = np.concatenate((pc_inbox_label, pc_outbox_label), axis=0)
    np.random.shuffle(pc_with_label)
    point_coords = pc_with_label[:, :2].tolist()
    point_label = pc_with_label[:, 2].tolist()
    return point_coords, point_label