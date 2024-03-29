import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear

from detectron2.layers import cat

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits) #torch.Size([125, 1, 176, 320])
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )#torch.Size([125, 1, 8, 176, 320])
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold 
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def compute_lidar_sim_loss(mask_logits, lidarin_coords):
    assert mask_logits.dim() == 4
    lidarin_gt_coords_wrt_img = get_point_coords_wrt_img(mask_logits.shape[2:], lidarin_coords.unsqueeze(dim=1))
    lidarin_logits = point_sample(
        mask_logits,
        lidarin_gt_coords_wrt_img,
        align_corners=False,
    )

    log_fg_prob = F.logsigmoid(lidarin_logits)
    log_bg_prob = F.logsigmoid(-lidarin_logits)

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    N = lidarin_coords.shape[-2]
    log_fg_prob_unfold = log_fg_prob.expand(-1,-1,N,-1)
    log_bg_prob_unfold = log_bg_prob.expand(-1,-1,N,-1)
    log_same_fg_prob = log_fg_prob + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob + log_bg_prob_unfold


    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    #the probability of making the same prediction = p_i * (1-p_j) + (1 - p_i) * p_j
    log_unsame_fg_prob = log_fg_prob + log_bg_prob_unfold
    log_unsame_bg_prob = log_bg_prob + log_fg_prob_unfold
    max_unsame = torch.max(log_unsame_fg_prob, log_unsame_bg_prob)
    log_unsame_prob = torch.log(
        torch.exp(log_unsame_fg_prob - max_unsame) +
        torch.exp(log_unsame_bg_prob - max_unsame)
    ) + max_unsame

    # loss = -log(prob)
    return -log_same_prob[:, 0], -log_unsame_prob[:, 0]


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.lidarsim_weight = cfg.MODEL.LIDARSIM_WEIGHT
        self.lidarsim_thresh = cfg.MODEL.LIDARSIM_THRESH
        self.pointloss_weight = cfg.MODEL.POINTLOSS_WEIGHT
        self.lidarsim_rgb_thresh = cfg.MODEL.LIDARSIM_RGB_THRESH
        self.lidarsim_rgb_weight = cfg.MODEL.LIDARSIM_RGB_WEIGHT
        self.LWSIS = cfg.INPUT.USE_3DPOINTS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.register_buffer("_iter", torch.zeros([1]))

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                if not self.boxinst_enabled:
                    losses["loss_mask"] = dummy_loss
                else:
                    losses["loss_prj"] = dummy_loss
                    losses["loss_pairwise"] = dummy_loss
            else:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_scores = mask_logits.sigmoid()

                if self.boxinst_enabled:
                    # box-supervised BoxInst losses
                    image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                    image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)

                    pairwise_losses = compute_pairwise_term(
                        mask_logits, self.pairwise_size,
                        self.pairwise_dilation
                    )

                    weights = (image_color_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
                    loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

                    warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                    loss_pairwise = loss_pairwise * warmup_factor

                    #dynamic weigth according to box size
                    gt_boxes = torch.cat([per_im.gt_boxes.tensor for per_im in gt_instances])
                    gt_areas = (gt_boxes[:,2]-gt_boxes[:,0]) * (gt_boxes[:,3]-gt_boxes[:,1])
                    gt_weight = torch.sqrt(gt_areas/self.areaRng[-1][0])
                    gt_weight_norm = (gt_weight - gt_weight.min()) / (gt_weight.max() - gt_weight.min() + 1e-5)
                    gt_weight = gt_weight[gt_inds]
                    gt_weight_norm = gt_weight_norm[gt_inds]

                    if self.LWSIS:
                        update_large = 0
                        if update_large:
                            gt_boxes = torch.cat([per_im.gt_boxes.tensor for per_im in gt_instances])
                            gt_areas = (gt_boxes[:,2]-gt_boxes[:,0]) * (gt_boxes[:,3]-gt_boxes[:,1])
                            gt_large =  (gt_areas >= self.areaRng[-1][0]) * (gt_areas <= self.areaRng[-1][1])
                            large_mask = torch.tensor([gt_large[x] for x in gt_inds])
                            gt_inds = gt_inds[large_mask]
                            mask_logits = mask_logits[large_mask]

                        gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
                        gt_coords = torch.cat([per_im.gt_point_coords for per_im in gt_instances])
                        gt_coords = gt_coords[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
                        gt_coords_wrt_img = get_point_coords_wrt_img(mask_logits.shape[2:], gt_coords)
                        gt_labels = torch.cat([per_im.gt_point_labels for per_im in gt_instances])
                        gt_labels = gt_labels[gt_inds].to(dtype=mask_feats.dtype)

                        point_logits = point_sample(
                            mask_logits,
                            gt_coords_wrt_img,
                            align_corners=False,
                        )
                        if len(gt_inds) == 0:
                            loss_lidar = point_logits.sum()*0
                        else:
                            loss_lidar = roi_mask_point_loss(point_logits, gt_labels, gt_weight_norm) #, 
                        losses["loss_lidar"] = loss_lidar * warmup_factor * self.pointloss_weight    

                    if self.LWSIS:
                        lidar_sim = torch.cat([x.gt_lidarsim for x in gt_instances])                      
                        lidar_coords = torch.cat([per_im.gt_lidarcoord for per_im in gt_instances])
                        lidar_sim = lidar_sim[gt_inds].to(dtype=mask_feats.dtype)
                        lidar_coords = lidar_coords[gt_inds].to(dtype=mask_feats.dtype)

                        gt_resnet_sim = torch.cat([x.gt_resnet_sim for x in gt_instances])
                        gt_resnet_sim = gt_resnet_sim[gt_inds].to(dtype=mask_feats.dtype)

                        lidar_sim_losses_same, _ = compute_lidar_sim_loss(mask_logits, lidar_coords)

                        weight1 = (gt_resnet_sim > self.lidarsim_rgb_thresh).float()
                        resnet_sim_losses = lidar_sim_losses_same * (1-gt_weight_norm).unsqueeze(1).unsqueeze(1)
                        resnet_sim_losses = (resnet_sim_losses * weight1).sum() / weight1.sum().clamp(min=1.0) 
                        losses["resnet_sim_losses"] = resnet_sim_losses * warmup_factor * self.lidarsim_rgb_weight

                        weight2 = (lidar_sim >= self.lidarsim_thresh).float()
                        lidar_sim_losses = lidar_sim_losses_same * (1-gt_weight_norm).unsqueeze(1).unsqueeze(1)
                        lidar_sim_losses = (lidar_sim_losses * weight2).sum() / weight2.sum().clamp(min=1.0)
                        losses["lidar_sim_losses"] = lidar_sim_losses  * warmup_factor * self.lidarsim_weight

                    losses.update({
                        "loss_prj": loss_prj_term,
                        "loss_pairwise": loss_pairwise,
                    })
                else:
                    # fully-supervised CondInst losses
                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                    losses["loss_mask"] = loss_mask

            return losses
        else:
            if len(pred_instances) > 0:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()

            return pred_instances


def roi_mask_point_loss(mask_logits, gt_labels, gt_weigth=None):
    """
    Compute the point-based loss for instance segmentation mask predictions
    given point-wise mask prediction and its corresponding point-wise labels.
    Args:
        mask_logits (Tensor): A tensor of shape (R, C, P) or (R, 1, P) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images, C is the
            number of foreground classes, and P is the number of points sampled for each mask.
            The values are logits.
        point_labels (Tensor): A tensor of shape (R, P), where R is the total number of
            predicted masks and P is the number of points for each mask.
            Labels with value of -1 will be ignored.
    Returns:
        point_loss (Tensor): A scalar tensor containing the loss.
    """
    point_ignores = gt_labels == -1
    if gt_weigth is not None:
        gt_weigth = (~point_ignores*gt_weigth.unsqueeze(1))
    else:
        gt_weigth = ~point_ignores
    mask_logits = mask_logits.squeeze(1).squeeze(1)
    point_loss = F.binary_cross_entropy_with_logits(
        mask_logits, gt_labels.to(dtype=torch.float32), weight=gt_weigth, reduction="mean"
    )
    return point_loss

def point_sample(input, point_coords, **kwargs):

    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """

    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    return output


def get_point_coords_from_point_annotation(instances):
    """
    Load point coords and their corresponding labels from point annotation.

    Args:
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
        point_labels (Tensor): A tensor of shape (N, P) that contains the labels of P
            sampled points. `point_labels` takes 3 possible values:
            - 0: the point belongs to background
            - 1: the point belongs to the object
            - -1: the point is ignored during training
    """
    point_coords_list = []
    point_labels_list = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        point_coords = instances_per_image.gt_point_coords.to(torch.float32)
        point_labels = instances_per_image.gt_point_labels.to(torch.float32).clone()
        proposal_boxes_per_image = instances_per_image.proposal_boxes.tensor

        # Convert point coordinate system, ground truth points are in image coord.
        point_coords_wrt_box = get_point_coords_wrt_box(proposal_boxes_per_image, point_coords)

        # Ignore points that are outside predicted boxes.
        point_ignores = (
            (point_coords_wrt_box[:, :, 0] < 0)
            | (point_coords_wrt_box[:, :, 0] > 1)
            | (point_coords_wrt_box[:, :, 1] < 0)
            | (point_coords_wrt_box[:, :, 1] > 1)
        )
        point_labels[point_ignores] = -1

        point_coords_list.append(point_coords_wrt_box)
        point_labels_list.append(point_labels)

    return (
        cat(point_coords_list, dim=0),
        cat(point_labels_list, dim=0),
    )

def get_point_coords_wrt_img(mask_size, point_coords):
    """
    Convert image-level absolute coordinates to box-normalized [0, 1] x [0, 1] point cooordinates.
    Args:
        boxes_coords (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
            coordinates.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    Returns:
        point_coords_wrt_box (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.
    """
    with torch.no_grad():
        point_coords_wrt_mask = point_coords.clone()
        point_coords_wrt_mask[:, :, :,  0] /= mask_size[1]
        point_coords_wrt_mask[:, :, :, 1] /= mask_size[0]
    return point_coords_wrt_mask