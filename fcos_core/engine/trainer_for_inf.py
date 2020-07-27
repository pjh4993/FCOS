# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
from torchvision import transforms as T
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict, (proposal, GT) = model(images, targets)

        size = [l_p.shape[2] * l_p.shape[3] for l_p in proposal ]

        for l in range(len(GT)):
            tmp_GT = GT[l].view(-1, 1)
            one_hot = (tmp_GT.cpu() == torch.arange(160).reshape(1,160)).float()
            one_hot = one_hot[:,::2] + one_hot[:,1::2]
            GT[l] = one_hot
        
        for l in range(len(proposal)):
            proposal[l] = torch.max(proposal[l].squeeze(0).permute(1,2,0), dim=2)[1].view(-1)
        
        for l in range(len(proposal)):
            tmp_GT = proposal[l]
            one_hot = (tmp_GT.unsqueeze(1).cpu() == torch.arange(160).reshape(1,160)).float()
            one_hot = one_hot[:,::2] + one_hot[:,1::2]
            one_hot = one_hot.reshape(size[l], -1)
            proposal[l] = one_hot
 


        PIXEL_MEAN =  [-102.9801, -115.9465, -122.7717]
        PIXEL_STD = [1.0, 1.0, 1.0]
        unnorm = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

        new_image = unnorm(images.tensors[0])
        new_image = new_image.cpu()[[2,1,0],:,:]/255
        cmaps= [
            'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        fpl = 8
        img_size = images.tensors[0].shape[1:]
        label = torch.unique(targets[0].get_field("labels"))
        up = torch.nn.UpsamplingNearest2d(size=(img_size[0],img_size[1]))
        heatmap_image, heat_axe = plt.subplots(nrows=2, ncols=5)

        heatmap_image.text(0,0,"".join(str(label.tolist())))
        for l in range(len(proposal)):
            fig_image, axe_image = plt.subplots(nrows=2, ncols=1)
            tmp_image = new_image.clone()
            axe_image[0].imshow(T.ToPILImage()(tmp_image))
            axe_image[1].imshow(T.ToPILImage()(tmp_image))
            y = 0
            heat = torch.zeros(tmp_image.shape[1],tmp_image.shape[2])
            small = torch.zeros(img_size[0]//fpl + (img_size[0]%fpl > 0), img_size[1]//fpl + (img_size[1]%fpl > 0))
            heat_GT = torch.zeros(tmp_image.shape[1],tmp_image.shape[2])
            small_GT = torch.zeros(img_size[0]//fpl + (img_size[0]%fpl > 0), img_size[1]//fpl + (img_size[1]%fpl > 0))

            small = torch.zeros(img_size[0]//fpl + (img_size[0]%fpl > 0), img_size[1]//fpl + (img_size[1]%fpl > 0))
            for ulb in range(len(label)):
                tmp_prop = proposal[l][:,label[ulb]].reshape(img_size[0]//fpl + (img_size[0]%fpl > 0), img_size[1]//fpl + (img_size[1]%fpl > 0))
                tmp_prop = tmp_prop.view(1,1,tmp_prop.shape[0], tmp_prop.shape[1]).detach()
                small += tmp_prop.view(tmp_prop.shape[2],tmp_prop.shape[3]).cpu()
                tmp_prop = up(tmp_prop).view(img_size[0], img_size[1])
                heat += tmp_prop.cpu()

                tmp_prop = GT[l][:,label[ulb]].reshape(img_size[0]//fpl + (img_size[0]%fpl > 0), img_size[1]//fpl + (img_size[1]%fpl > 0))
                tmp_prop = tmp_prop.view(1,1,tmp_prop.shape[0], tmp_prop.shape[1]).detach()
                small_GT += tmp_prop.view(tmp_prop.shape[2],tmp_prop.shape[3]).cpu()
                tmp_prop = up(tmp_prop).view(img_size[0], img_size[1])
                heat_GT += tmp_prop.cpu()
               
                #axe_image.text(0, y, cmaps[ulb%len(cmaps)] + str(label[ulb]) + str(tmp_prop.sum()), fontsize=12)
                #y = y+12

            heat_axe[0,l].matshow(small.cpu().numpy())
            heat_axe[1,l].matshow(small_GT.cpu().numpy())
            axe_image[0].matshow(heat.cpu().numpy(), alpha=0.5)
            axe_image[1].matshow(heat_GT.cpu().numpy(), alpha=0.5)
            fpl = fpl*2
            fig_image.savefig('test' + str(l)+'.jpg')
        heatmap_image.savefig('heat.jpg')
        exit()
        
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
