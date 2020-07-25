import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_counter_postprocessor
from .loss import make_counter_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d


class CounterHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CounterHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        num_classes = self.num_classes
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        cnt_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            
            cnt_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cnt_tower.append(nn.GroupNorm(32, in_channels))
            cnt_tower.append(nn.ReLU())
            

        cnt_tower.append(
            nn.Conv2d(
                in_channels, num_classes * 8, kernel_size=3,stride=1,padding=1,bias=True
            )
        )
        cnt_tower.append(nn.GroupNorm(8, num_classes*8))
        cnt_tower.append(nn.ReLU())

        self.add_module('cnt_tower', nn.Sequential(*cnt_tower))

        self.cnt_regression = nn.Conv2d(
            num_classes * 8, num_classes * 8, kernel_size=3,stride=1,padding=1
        )
        

        # initialization
        for modules in [self.cnt_tower,
                        self.cnt_regression]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        """
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cnt_regression.bias, bias_value)
        """

    def forward(self, x):
        counted = []
        for l, feature in enumerate(x):
            cnt_tower = self.cnt_tower(feature)
            cnt_regressed = self.cnt_regression(cnt_tower)
            cnt_regressed = torch.pow(cnt_regressed , 2)
            cnt_regressed_per_class = list(torch.split(cnt_regressed, 8, dim=1))
            counted_per_level = []
            for cnt in cnt_regressed_per_class:
                counted_per_level.append(cnt.sum(1).unsqueeze(1))
            counted_per_level = torch.cat(counted_per_level, dim=1)
            counted.append(counted_per_level)
       
        return counted


class CounterModule(torch.nn.Module):
    """
    Module for Counter computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(CounterModule, self).__init__()

        head = CounterHead(cfg, in_channels)

        count_test = make_counter_postprocessor(cfg)

        loss_evaluator = make_counter_loss_evaluator(cfg)
        self.head = head
        self.count_test = count_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        obj_count = self.head(features)
        locations = self.compute_locations(features)
        self.images = images
 
        if self.training:
            return self._forward_train(
                locations, obj_count, targets
            )
        else:
            return self._forward_test(
                locations, obj_count, images.image_sizes
            )

    def _forward_train(self, locations, obj_count, targets):
        loss_count = self.loss_evaluator(
            locations, obj_count, targets
        )
        losses = {
            "loss_count": loss_count
        }
        #reg_targets = self.loss_evaluator.prepare_targets(locations, targets)
        return None, losses

    def _forward_test(self, locations, obj_count, image_sizes):
        """
        count = self.count_test(
            locations, obj_count, image_sizes
        )
        """
        return obj_count, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_counter(cfg, in_channels):
    return CounterModule(cfg, in_channels)
