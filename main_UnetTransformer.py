import os, torch, argparse, sys, math
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Unet, self).__init__()

        self.down_conv_1 = self.conv_block(in_channels, out_channels)
        self.down_conv_2 = self.conv_block(out_channels, out_channels * 2)
        self.down_conv_3 = self.conv_block(out_channels * 2, out_channels * 4)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # Encoder (downsampling)
        conv1 = self.down_conv_1(x)
        x = self.maxpool(conv1)
        conv2 = self.down_conv_2(x)
        x = self.maxpool(conv2)
        conv3 = self.down_conv_3(x)

        return conv3

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


class AccidentDetector(nn.Module):
    def __init__(self, num_features=256, num_classes=2, num_layers=3, num_heads=8, dropout=0.3):
        super(AccidentDetector, self).__init__()
        self.num_features = num_features

        self.encoder = Unet()

        self.reduce_dim = nn.Linear(num_features * 56 * 56, 256)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=num_features, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        batch_size, seq_length = x.size(0), x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))

        features = self.encoder(x)
        features = features.view(batch_size, seq_length, self.num_features * 56 * 56)

        features = self.reduce_dim(features)
        features = features.permute(1, 0, 2)

        memory = torch.zeros_like(features)
        output = self.decoder(features, memory)
        output = output.permute(1, 0, 2)

        output = self.fc(output)

        return output


def train(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = 0
    accu_num = 0
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        masks, labels = data
        sample_num += masks.shape[0]

        masks = masks.unsqueeze(1)
        masks = masks.to(device)
        labels = labels.to(device)

        pred = model(masks)

        pred_classes = pred.argmax(dim=2)
        labels = labels.unsqueeze(1).repeat(1, pred.shape[1])

        accu_num += torch.eq(pred_classes, labels).sum().item()

        loss = loss_function(pred.transpose(1, 2), labels)
        loss.backward()
        accu_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss / (step + 1),
                                                                               accu_num / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss / (step + 1), accu_num / sample_num


def evaluate(model, data_loader, device, epoch):
    model.eval()

    loss_function = torch.nn.CrossEntropyLoss()

    accu_num = 0
    accu_loss = 0
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        masks, labels = data
        sample_num += masks.shape[0]

        masks = masks.unsqueeze(1)
        masks = masks.to(device)
        labels = labels.to(device)

        pred = model(masks)
        pred_classes = torch.max(pred, dim=2)[1]

        labels = labels.unsqueeze(1).repeat(1, pred.shape[1])

        accu_num += torch.eq(pred_classes, labels).sum().item()

        loss = loss_function(pred.transpose(1, 2), labels)

        accu_loss += loss.item()

        pred_labels = pred_classes.view(-1).cpu().numpy()
        true_labels = labels.view(-1).cpu().numpy()

        precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=1)
        recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=1)
        f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=1)
        accuracy = accuracy_score(true_labels, pred_labels)

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss / (step + 1),
                                                                               accu_num / sample_num)

    return accu_loss / (step + 1), accu_num / sample_num, precision, recall, f1, accuracy


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    tb_writer = SummaryWriter("././runs/")
    pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    amount_ac = len(os.listdir(args.dataset_dir + '/accident'))
    amount_nac = len(os.listdir(args.dataset_dir + '/no_accident'))
    print('accident images : %d, no_accident images : %d' % (amount_ac, amount_nac))

    train_dataset = datasets.ImageFolder(root=args.dataset_dir, transform=pipeline)
    train_loader = DataLoader(train_dataset, batch_size=amount_ac + amount_nac)

    images, labels = next(iter(train_loader))

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    x_train, y_train, x_label, y_label = train_test_split(images, labels, test_size=0.25)
    train_loader = DataLoader(TensorDataset(x_train, x_label), batch_size=args.batch_size,
                              pin_memory=True, num_workers=nw, shuffle=True)
    val_loader = DataLoader(TensorDataset(y_train, y_label), batch_size=args.batch_size,
                            pin_memory=True, num_workers=nw, shuffle=True)

    model = nn.DataParallel(AccidentDetector().to(device), device_ids=[0, 1])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        train_loss, train_acc = train(model=model, optimizer=optimizer, data_loader=train_loader, device=device,
                                      epoch=epoch)
        val_loss, val_acc, precision, recall, f1, accuracy = evaluate(model=model, data_loader=val_loader,
                                                                      device=device, epoch=epoch)

        scheduler.step()

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "precision", "recall", "f1",
                "accuracy"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], precision, epoch)
        tb_writer.add_scalar(tags[6], recall, epoch)
        tb_writer.add_scalar(tags[7], f1, epoch)
        tb_writer.add_scalar(tags[8], accuracy, epoch)
        torch.save(model.state_dict(), "././model/Unet-Transformer.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset-dir', type=str, default='././dataset/bboxes_mask_label')

    opt = parser.parse_args()


    main(opt)