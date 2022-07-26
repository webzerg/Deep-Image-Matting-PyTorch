import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from PIL import Image
import os

from config import device, im_size, grad_clip, print_freq
from data_gen import DIMDataset
from models import DIMModel
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, get_learning_rate, \
    alpha_prediction_loss, adjust_learning_rate


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0
    decays_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = DIMModel(n_classes=3, in_channels=3, is_unpooling=True, pretrain=True)
        model = nn.DataParallel(model)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model'].module
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = DIMDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_dataset = DIMDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        if args.optimizer == 'sgd' and epochs_since_improvement == 10:
            break

        if args.optimizer == 'sgd' and epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
            checkpoint = 'BEST_checkpoint_weight.tar'
            checkpoint = torch.load(checkpoint)
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']
            decays_since_improvement += 1
            print("\nDecays since last improvement: %d\n" % (decays_since_improvement,))
            adjust_learning_rate(optimizer, 0.6 ** decays_since_improvement)

        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)
        effective_lr = get_learning_rate(optimizer)
        print('Current effective learning rate: {}\n'.format(effective_lr))

        writer.add_scalar('Train_Loss', train_loss, epoch)
        writer.add_scalar('Learning_Rate', effective_lr, epoch)

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           logger=logger,
                           epoch=epoch)

        writer.add_scalar('Valid_Loss', valid_loss, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            decays_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)

def trimap_loss(pred_trimap, gt_trimap):
    class_weights = torch.tensor([1.0, 3.0, 1.0], dtype=torch.float).to(device)
    loss = nn.CrossEntropyLoss(weight=class_weights)
    # pred_vals = pred_trimap[:, 0, :]
    # gt_vals = gt_trimap[:, 1, :]
    return loss(pred_trimap, gt_trimap)

epoch_result_dir = './out_result_weight/'

def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (img, alpha_label) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device)  # [N, 4, 320, 320]
        gt_alpha = (alpha_label[:, 0, :, :].unsqueeze(1)).type(torch.FloatTensor).to(device)  # in:32*2*320*320;  [N, 320, 320]
        gt_trimap = alpha_label[:, 1, :, :].type(torch.LongTensor).to(device)  # in:32*2*320*320;  [N, 320, 320]

        # save the label input image
        image_name = 'alpha_epoch_' + str(epoch) + '_iteration_' + str(i) + '_input_.jpg'
        image_raw = img.detach().cpu().numpy()[0, 0:3, :, :]
        image_data = (image_raw*255).astype(np.uint8)
        Image.fromarray(image_data.transpose(1, 2, 0), 'RGB').save(
            os.path.join(epoch_result_dir, image_name)
        )

        # # save the output trimap image
        # image_name = 'alpha_epoch_' + str(epoch) + '_iteration_' + str(i) + '_trimap_.jpg'
        # image_raw = img.detach().cpu().numpy()[0, 3, :, :]
        # image_data = (image_raw*255).astype(np.uint8)
        # Image.fromarray(image_data).save(
        #     os.path.join(epoch_result_dir, image_name)
        # )

        # save the label alpha image
        image_name = 'alpha_epoch_' + str(epoch) + '_iteration_' + str(i) + '_label_.jpg'
        image_raw = gt_alpha.detach().cpu().numpy()[0, 0, :, :]
        image_data = (image_raw*255).astype(np.uint8)
        Image.fromarray(image_data).save(
            os.path.join(epoch_result_dir, image_name)
        )

        # save the tripmap GT image
        image_name = 'trimap_epoch_' + str(epoch) + '_iteration_' + str(i) + '_label_.jpg'
        image_raw = gt_trimap.detach().cpu().numpy()[0, :, :]
        maskGT = np.zeros(image_raw.shape)
        maskGT.fill(127)
        maskGT[image_raw <= 0] = 0
        maskGT[image_raw >= 2] = 255
        image_data = maskGT.astype(np.uint8)
        Image.fromarray(image_data).save(
            os.path.join(epoch_result_dir, image_name)
        )

        alpha_label = alpha_label.reshape((-1, 2, im_size * im_size))  # out: 32*2*102400; [N, 320*320]
        # Forward prop.
        # alpha_out = model(img)  # In: [N, 3, 320, 320]
        trimap_out = model(img)  # In: [N, 3, 320, 320]

        # save the out trimap image: trimap_out is N,3,320,320
        trimap_argmax = trimap_out.argmax(dim=1)
        image_name = 'trimap_epoch_' + str(epoch) + '_iteration_' + str(i) + '_out_.jpg'
        image_raw = trimap_argmax.detach().cpu().numpy()[0, :, :]  #just plot first dim
        maskOut = np.zeros(image_raw.shape)
        maskOut.fill(127)
        maskOut[image_raw <= 0] = 0
        maskOut[image_raw >= 2] = 255
        image_data = maskOut.astype(np.uint8)
        Image.fromarray(image_data).save(
            os.path.join(epoch_result_dir, image_name)
        )

        trimap_out = trimap_out.reshape((-1, 3, im_size * im_size))  # In: 32*320*320, out: 32*1*102400, old out: [N, 320*320]
        gt_trimap_flat = gt_trimap.reshape((-1, im_size * im_size))

        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        # loss = alpha_prediction_loss(alpha_out, alpha_label)
        loss = trimap_loss(trimap_out, gt_trimap_flat) #alpha_label is 2 rows (alpha and mask(trimap))

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status

        if i % print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses)
            logger.info(status)

    return losses.avg


def valid(valid_loader, model, logger, epoch):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()

    # Batches
    for img, alpha_label in valid_loader:
        # Move to GPU, if available
        img = img.type(torch.FloatTensor).to(device)  # [N, 3, 320, 320]
        # alpha_label = alpha_label.type(torch.FloatTensor).to(device)  # [N, 320, 320]
        # alpha_label = alpha_label.reshape((-1, 2, im_size * im_size))  # [N, 320*320]
        gt_alpha = (alpha_label[:, 0, :, :].unsqueeze(1)).type(torch.FloatTensor).to(device)  # in:32*2*320*320;  [N, 320, 320]
        gt_trimap = alpha_label[:, 1, :, :].type(torch.LongTensor).to(device)  # in:32*2*320*320;  [N, 320, 320]

        for i in range(6):
            # save the label input image
            image_name = 'valid_alpha_epoch_' + str(epoch) + '_img_' + str(i) + '_input_.jpg'
            image_raw = img.detach().cpu().numpy()[i, 0:3, :, :]
            image_data = (image_raw*255).astype(np.uint8)
            Image.fromarray(image_data.transpose(1, 2, 0), 'RGB').save(
                os.path.join(epoch_result_dir, image_name)
            )

            # save the label alpha image
            image_name = 'valid_alpha_epoch_' + str(epoch) + '_img_' + str(i) + '_label_.jpg'
            image_raw = gt_alpha.detach().cpu().numpy()[i, 0, :, :]
            image_data = (image_raw*255).astype(np.uint8)
            Image.fromarray(image_data).save(
                os.path.join(epoch_result_dir, image_name)
            )

            # save the tripmap GT image
            image_name = 'valid_trimap_epoch_' + str(epoch) + '_img_' + str(i) + '_label_.jpg'
            image_raw = gt_trimap.detach().cpu().numpy()[i, :, :]
            maskGT = np.zeros(image_raw.shape)
            maskGT.fill(127)
            maskGT[image_raw <= 0] = 0
            maskGT[image_raw >= 2] = 255
            image_data = maskGT.astype(np.uint8)
            Image.fromarray(image_data).save(
                os.path.join(epoch_result_dir, image_name)
            )

        # Forward prop.
        # alpha_out = model(img)  # [N, 320, 320]
        # alpha_out = alpha_out.reshape((-1, 1, im_size * im_size))  # [N, 320*320]

        trimap_out = model(img)  # [N, 3, 320, 320]

        for j in range(7):
            # save the out trimap image: trimap_out is N,3,320,320
            trimap_argmax = trimap_out.argmax(dim=1)
            image_name = 'valid_trimap_epoch_' + str(epoch) + '_img_' + str(j) + '_out_.jpg'
            image_raw = trimap_argmax.detach().cpu().numpy()[j, :, :]  #just plot first dim
            maskOut = np.zeros(image_raw.shape)
            maskOut.fill(127)
            maskOut[image_raw <= 0] = 0
            maskOut[image_raw >= 2] = 255
            image_data = maskOut.astype(np.uint8)
            Image.fromarray(image_data).save(
                os.path.join(epoch_result_dir, image_name)
            )

        trimap_out = trimap_out.reshape((-1, 3, im_size * im_size))  # In: 32*320*320, out: 32*1*102400, old out: [N, 320*320]
        gt_trimap_flat = gt_trimap.reshape((-1, im_size * im_size))

        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        # loss = alpha_prediction_loss(alpha_out, alpha_label)
        loss = trimap_loss(trimap_out, gt_trimap_flat) #alpha_label is 2 rows (alpha and mask(trimap))

        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        # loss = alpha_prediction_loss(alpha_out, alpha_label)

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    status = 'Validation: Loss {loss.avg:.4f}\n'.format(loss=losses)

    logger.info(status)

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
