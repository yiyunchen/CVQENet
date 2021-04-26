from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_load import get_test_set
import torch.backends.cudnn as cudnn
import cv2
import sys
import datetime
from utils import Logger, calculate_ssim, calculate_psnr
import time
# from model_arch.arch_model import VQE
from model_arch.arch import VQE
from model_arch.arch_model import VQEO
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(description='PyTorch VQE Example')
# network setting
parser.add_argument('--scale', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--in_channel', type=int, default=3, help="the channel number of input image")
parser.add_argument('--nf', type=int, default=64, help='the channel number of feature')
parser.add_argument('--nb1', type=int, default=10, help='the nb of ff')
parser.add_argument('--nb2', type=int, default=20, help='the nb of qe')
# data loader parameters
parser.add_argument('--testbatchsize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
# testing setting
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=123')
parser.add_argument('--gpus', default='5', type=str, help='gpu ids (default: 0)')
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--pretrain', type=str, default='../pretrainModel/CVQENet.pth') 
parser.add_argument('--test_dir', type=str, default='/media/data4/vqe/NTIRE/Track1_release_test_data')  # /media/data4/vqe/NTIRE/Track1_release_test_data
# test saving setting
parser.add_argument('--save_test_log', type=str, default='../results/log')
parser.add_argument('--image_out', type=str, default='../results/')

opt = parser.parse_args()
# gpus_list = range(opt.gpus)
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
print(opt)


def main():
    sys.stdout = Logger(os.path.join(opt.save_test_log, 'VQE_LD_Final_' + systime + '.txt'))
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = False
        torch.cuda.manual_seed(opt.seed)
    pin_memory = True if use_gpu else False
    n_b1 = opt.nb1
    n_b2 = opt.nb2
    # dunet = DUNet(opt.in_channel, n_c, n_b1, n_b2)  # initial filter generate network

    net = VQEO(opt.nf, n_b1, n_b2)
    print(net)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in net.parameters()) * 4 / 1048576))
    print('===> {}L model has been initialized'.format(n_b1 + n_b2))
    net = torch.nn.DataParallel(net)
    print('===> load pretrained model')
    if os.path.isfile(opt.pretrain):
        net.load_state_dict(torch.load(opt.pretrain, map_location=lambda storage, loc: storage))
        print('===> pretrained model is load')
    else:
        raise Exception('pretrain model is not exists')
    if use_gpu:
        net = net.cuda()

    # print('===> Loading test Datasets')
    # PSNR_avg = 0
    # SSIM_avg = 0
    # count = 0
    # scene_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020']  # Vid4
    # for scene_name in scene_list:
    #    test_set = get_test_set(opt.test_dir, opt.scale, scene_name)
    #    # print(test_set)
    #    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchsize, shuffle=False,
    #                            pin_memory=pin_memory, drop_last=False)
    #    print('===> DataLoading Finished')
    #    PSNR, SSIM = test(test_loader, net, opt.scale, scene_name)
    #    PSNR_avg += PSNR
    #    SSIM_avg += SSIM
    #    count += 1
    # PSNR_avg = PSNR_avg / len(scene_list)
    # SSIM_avg = SSIM_avg / len(scene_list)
    # print('==> Average PSNR = {:.6f}'.format(PSNR_avg))
    # print('==> Average SSIM = {:.6f}'.format(SSIM_avg))


    # real data testing
    print('===> Loading test Datasets')
    scene_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']  # Vid4

    for scene_name in scene_list:
        test_set = get_test_set(opt.test_dir, opt.scale, scene_name)
        test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchsize, shuffle=False,
                                  pin_memory=pin_memory, drop_last=False)
        print('===> DataLoading Finished')
        real_data_test(test_loader, net, opt.scale, scene_name)


def test_ensamble(LH, op):
    # print(type(LH))
    npLH = LH.cpu().numpy()
    if op == 'v':
        tfLH = npLH[:, :, :, :, ::-1].copy()
    elif op == 'h':
        tfLH = npLH[:, :, :, ::-1, :].copy()
    elif op == 't':
        tfLH = npLH.transpose((0, 1, 2, 4, 3)).copy()

    en_LH = torch.Tensor(tfLH).cuda()
    return en_LH


def test_ensamble_2(LH, op):
    npLH = LH.cpu().numpy()
    if op == 'v':
        tfLH = npLH[:, :, :, ::-1].copy()
    elif op == 'h':
        tfLH = npLH[:, :, ::-1, :].copy()
    elif op == 't':
        tfLH = npLH.transpose((0, 1, 3, 2)).copy()

    en_LH = torch.Tensor(tfLH).cuda()
    return en_LH


def test(test_loader, net, scale, scene_name):
    train_mode = False
    net.eval()
    count = 0
    PSNR = 0
    SSIM = 0
    PSNR_t = 0
    SSIM_t = 0
    out = []
    for image_num, data in enumerate(test_loader):
        x_input, target = data[0], data[1]
        # print(x_input.shape)
        B, _, _, _, _ = x_input.shape
        with torch.no_grad():
            x_input = Variable(x_input).cuda()
            target = Variable(target).cuda()
            t0 = time.time()
            # prediction = net(x_input)
            # ensamble test
            if True:
                # x_input = Variable(x_input).cuda()
                x_input_list = [x_input]
                for tf in ['v', 'h', 't']:
                    x_input_list.extend([test_ensamble(t, tf) for t in x_input_list])
                prediction_list = [net(aug) for aug in x_input_list]
                for i in range(len(prediction_list)):
                    if i > 3:
                        prediction_list[i] = test_ensamble_2(prediction_list[i], 't')
                    if i % 4 > 1:
                        prediction_list[i] = test_ensamble_2(prediction_list[i], 'h')
                    if (i % 4) % 2 == 1:
                        prediction_list[i] = test_ensamble_2(prediction_list[i], 'v')
                prediction_cat = torch.cat(prediction_list, dim=0)
                prediction = prediction_cat.mean(dim=0, keepdim=True)

        torch.cuda.synchronize()
        t1 = time.time()
        print("===> Timer: %.4f sec." % (t1 - t0))
        prediction = prediction.unsqueeze(2)
        count += 1
        prediction = prediction.squeeze(0).permute(1, 2, 3, 0)  # [T,H,W,C]
        prediction = prediction.cpu().numpy()[:, :, :, ::-1]  # tensor -> numpy, rgb -> bgr
        target = target.squeeze(0).permute(1, 2, 3, 0)  # [T,H,W,C]
        target = target.cpu().numpy()[:, :, :, ::-1]  # tensor -> numpy, rgb -> bgr

        save_img(prediction[0], scene_name, 2)
        # test_Y______________________
        # prediction_Y = bgr2ycbcr(prediction[0])
        # target_Y = bgr2ycbcr(target[0])
        # prediction_Y = prediction_Y * 255
        # target_Y = target_Y * 255
        # test_RGB _______________________________
        prediction_Y = prediction[0] * 255
        target_Y = target[0] * 255
        # ________________________________
        # calculate PSNR and SSIM
        print('PSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(calculate_psnr(prediction_Y, target_Y),
                                                       calculate_ssim(prediction_Y, target_Y)))
        PSNR += calculate_psnr(prediction_Y, target_Y)
        SSIM += calculate_ssim(prediction_Y, target_Y)
        out.append(calculate_psnr(prediction_Y, target_Y))

        print('===>{} PSNR = {}'.format(scene_name, PSNR))
        print('===>{} SSIM = {}'.format(scene_name, SSIM))
        PSNR_t = PSNR
        SSIM_t = SSIM

    return PSNR_t, SSIM_t


def real_data_test(test_loader, net, scale, scene_name):
    train_mode = False
    net.eval()
    # count = 0
    # out = []

    average_time = 0

    flag_num = 0

    for image_num, data in enumerate(test_loader):
        flag_num += 1

        x_input = data
        # B, _, T, _ ,_ = x_input.shape
        with torch.no_grad():
            x_input = Variable(x_input).cuda()
            t0 = time.time()
            # prediction = dunet(x_input)
            if True:
                # x_input = Variable(x_input).cuda()
                x_input_list = [x_input]
                for tf in ['v', 'h', 't']:
                    x_input_list.extend([test_ensamble(t, tf) for t in x_input_list])
                prediction_list = [net(aug) for aug in x_input_list]
                for i in range(len(prediction_list)):
                    if i > 3:
                        prediction_list[i] = test_ensamble_2(prediction_list[i], 't')
                    if i % 4 > 1:
                        prediction_list[i] = test_ensamble_2(prediction_list[i], 'h')
                    if (i % 4) % 2 == 1:
                        prediction_list[i] = test_ensamble_2(prediction_list[i], 'v')
                prediction_cat = torch.cat(prediction_list, dim=0)
                prediction = prediction_cat.mean(dim=0, keepdim=True)

        torch.cuda.synchronize()
        t1 = time.time()
        print("===> Timer: %.4f sec." % (t1 - t0))
        # prediction = torch.stack(out, dim=2)
        prediction = prediction.unsqueeze(2)
        prediction = prediction.squeeze(0).permute(1, 2, 3, 0)  # [T,H,W,C]
        prediction = prediction.cpu().numpy()[:, :, :, ::-1]  # tensor -> numpy, rgb -> bgr
        save_img(prediction[0], scene_name, image_num)

        average_time += (t1 - t0)

    print("time: {:.4f}".format(average_time / flag_num))



def save_img(prediction, scene_name, image_num):
    #save_dir = os.path.join(opt.image_out, systime)
    save_dir = os.path.join(opt.image_out, scene_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #image_dir = os.path.join(save_dir, '{}_{:03}'.format(scene_name, image_num + 1) + '.png')
    image_dir = os.path.join(save_dir, '{:03}'.format(image_num + 1) + '.png')
    cv2.imwrite(image_dir, prediction * 255, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    main()
