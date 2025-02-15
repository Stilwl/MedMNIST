import os
import argparse
from tqdm import trange
import numpy as np
import torch,torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import subprocess

from medmnist.models import ResNet18, ResNet50
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO


def main(args):
    ''' main function
    :param flag: name of subset

    '''

    flag_to_class = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNISTAxial,
        "organmnist_coronal": OrganMNISTCoronal,
        "organmnist_sagittal": OrganMNISTSagittal,
    }
    args.data_name = args.data_name.lower()
    args.net = args.net.lower()
    DataClass = flag_to_class[args.data_name]

    info = INFO[args.data_name]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    # start_epoch = 0
    lr = args.lr
    batch_size = args.bs
    # val_auc_list = []
    dir_path = os.path.join(args.output_root, '%s_checkpoints' % (args.data_name))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data_path = os.path.join(dir_path, 'save_data.txt')
    if not os.path.isfile(data_path):
        os.system(r"touch {}".format(data_path))#调用系统命令行来创建文件

    print('==> Preparing data...')
    train_transform = transforms.Compose(
        [transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    val_transform = transforms.Compose(
        [transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    test_transform = transforms.Compose(
        [transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = DataClass(root=args.input_root,
                                    split='train',
                                    transform=train_transform,
                                    download=args.download)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    val_dataset = DataClass(root=args.input_root,
                                  split='val',
                                  transform=val_transform,
                                  download=args.download)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    test_dataset = DataClass(root=args.input_root,
                                   split='test',
                                   transform=test_transform,
                                   download=args.download)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    # 可视化四张图片
    if args.verbose:
        datashow(train_loader)

    print('==> Building and training model...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.net=='resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device)
    else:
        model = ResNet18(in_channels=n_channels, num_classes=n_classes).to(device)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if args.optim=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(args.start_epoch, args.end_epoch):
        if epoch>=1:
            save_data = np.loadtxt(data_path)
            save_data = np.reshape(save_data,(-1,3))
            val_auc_list = save_data[:, 1]
            model_path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch-1, val_auc_list[epoch-1]))
            model.load_state_dict(torch.load(model_path)['net'])
        train(model, optimizer, criterion, train_loader, device, task)
        val(model, val_loader, device, task, dir_path, epoch)

    save_data = np.loadtxt(data_path)
    auc_list = save_data[:, 1]
    index = auc_list.argmax()
    print('epoch %s is the best model' % (index))

    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model, 'train', train_loader, device, task, args)
    test(model, 'val', val_loader, device, task, args)
    test(model, 'test', test_loader, device, task, args)

# 查看数据（可视化数据）
def datashow(train_loader):
    # 可视化样本，下面是输出了训练集中前4个样本
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
    ax = ax.flatten()
    images, label = next(iter(train_loader))
    for i in range(4):
        img = images[i].reshape(32, 32)
        # ax[i].imshow(img,cmap='Greys')
        ax[i].imshow(img)
    plt.tight_layout()
    plt.show()

def train(model, optimizer, criterion, train_loader, device, task):
    ''' training function
    :param model: the model to train
    :param optimizer: optimizer used in training
    :param criterion: loss function
    :param train_loader: DataLoader of training set
    :param device: cpu or cuda
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long().to(device)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()


def val(model, val_loader, device, task, dir_path, epoch):
    ''' validation function
    :param model: the model to validate
    :param val_loader: DataLoader of validation set
    :param device: cpu or cuda
    :param val_auc_list: the list to save AUC score of each epoch
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    :param dir_path: where to save model
    :param epoch: current epoch

    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)

    data_file = open(os.path.join(dir_path, 'save_data.txt'), 'a')
    data_file.write(str(epoch) + ' ' + str(auc) + ' ' + str(acc) + "\n")
    data_file.close()


def test(model, split, data_loader, device, task, args):
    ''' testing function
    :param model: the model to test
    :param split: the data to test, 'train/val/test'
    :param data_loader: DataLoader of data
    :param device: cpu or cuda
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    '''

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if args.output_root is not None:
            output_dir = os.path.join(args.output_root, args.data_name)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name',
                        default='breastmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--input_root',
                        default='./input',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    # parser.add_argument('--num_epoch',
    #                     default=100,
    #                     help='num of epochs of training',
    #                     type=int)
    parser.add_argument('--start_epoch',
                        default=0,
                        help='num of epochs of starting train',
                        type=int)
    parser.add_argument('--end_epoch',
                        default=100,
                        help='num of epochs of ending train',
                        type=int)
    parser.add_argument('--download',
                        default=True,
                        help='whether download the dataset or not',
                        type=bool)
    parser.add_argument('--lr',
                        type=float, 
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--bs',
                        type=int, 
                        default=128,
                        help='batch size')
    parser.add_argument('--net',
                        type=str, 
                        default='resnet18',
                        help='ResNet name')
    parser.add_argument('--optim',
                        default='Adam',
                        help='The optimizer',
                        type=str)
    parser.add_argument('--verbose',
                        default=False,
                        help='Show part of the dataset',
                        type=bool)

    args = parser.parse_args()
    main(args)
