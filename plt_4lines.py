import matplotlib.pyplot as plt
import csv

'''读取csv文件, 画图前需要删除csv中的第一行'''

def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append(float(row[2]))
        x.append(float(row[1]))
    return x, y

def train_acc_compared():
    plt.figure(1)
    x1, y1 = readcsv('./results/with_bs/train_acc.csv')
    plt.plot(x1, y1, color='#EDB120', label='The proposed framework', linewidth=2)
    x2, y2 = readcsv('./results/DCNN/train_acc.csv')
    plt.plot(x2, y2, color='#7E2F8E', label='DCNN', linewidth=2)
    x3, y3 = readcsv('./results/LSTMDTR/train_acc.csv')
    plt.plot(x3, y3, color='#77AC30', label='LSTMDTR', linewidth=2)
    x4, y4 = readcsv('./results/ViT-TA/train_acc.csv')
    plt.plot(x4, y4, color='#D95319', label='ViT-TA', linewidth=2)

    plt.xlabel('Epochs', fontdict={'family': 'serif', 'size': 20})
    plt.ylabel('Loss', fontdict={'family': 'serif', 'size': 20})
    # plt.title('The loss of CNN', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2,
               borderaxespad=0, prop = {'family': 'serif', 'size': 16})

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    plt.savefig('./results/train_acc_compared.jpg')
    plt.show()

def val_acc_compared():
    plt.figure(1)
    x1, y1 = readcsv('./results/with_bs/val_acc.csv')
    plt.plot(x1, y1, color='#EDB120', label='The proposed framework', linewidth=2)
    x2, y2 = readcsv('./results/DCNN/val_acc.csv')
    plt.plot(x2, y2, color='#7E2F8E', label='DCNN', linewidth=2)
    x3, y3 = readcsv('./results/LSTMDTR/val_acc.csv')
    plt.plot(x3, y3, color='#77AC30', label='LSTMDTR', linewidth=2)
    x4, y4 = readcsv('./results/ViT-TA/val_acc.csv')
    plt.plot(x4, y4, color='#D95319', label='ViT-TA', linewidth=2)

    plt.xlabel('Epochs', fontdict={'family': 'serif', 'size': 20})
    plt.ylabel('Loss', fontdict={'family': 'serif', 'size': 20})
    # plt.title('The loss of CNN', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2,
               borderaxespad=0, prop={'family': 'serif', 'size': 16})

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig('./results/val_acc_compared.jpg')
    plt.show()

def train_loss_compared():
    plt.figure(1)
    x1, y1 = readcsv('./results/with_bs/train_loss.csv')
    plt.plot(x1, y1, color='#EDB120', label='The proposed framework', linewidth=2)
    x2, y2 = readcsv('./results/DCNN/train_loss.csv')
    plt.plot(x2, y2, color='#7E2F8E', label='DCNN', linewidth=2)
    x3, y3 = readcsv('./results/LSTMDTR/train_loss.csv')
    plt.plot(x3, y3, color='#77AC30', label='LSTMDTR', linewidth=2)
    x4, y4 = readcsv('./results/ViT-TA/train_loss.csv')
    plt.plot(x4, y4, color='#D95319', label='ViT-TA', linewidth=2)

    plt.xlabel('Epochs', fontdict={'family': 'serif', 'size': 20})
    plt.ylabel('Loss', fontdict={'family': 'serif', 'size': 20})
    # plt.title('The loss of CNN', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2,
               borderaxespad=0, prop={'family': 'serif', 'size': 16})

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig('./results/train_loss_compared.jpg')
    plt.show()

def val_loss_compared():
    plt.figure(1)
    x1, y1 = readcsv('./results/with_bs/val_loss.csv')
    plt.plot(x1, y1, color='#EDB120', label='The proposed framework', linewidth=2)
    x2, y2 = readcsv('./results/DCNN/val_loss.csv')
    plt.plot(x2, y2, color='#7E2F8E', label='DCNN', linewidth=2)
    x3, y3 = readcsv('./results/LSTMDTR/val_loss.csv')
    plt.plot(x3, y3, color='#77AC30', label='LSTMDTR', linewidth=2)
    x4, y4 = readcsv('./results/ViT-TA/val_loss.csv')
    plt.plot(x4, y4, color='#D95319', label='ViT-TA', linewidth=2)

    plt.xlabel('Epochs', fontdict={'family': 'serif', 'size': 20})
    plt.ylabel('Loss', fontdict={'family': 'serif', 'size': 20})
    # plt.title('The loss of CNN', fontsize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2,
               borderaxespad=0, prop={'family': 'serif', 'size': 16})

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig('./results/val_loss_compared.jpg')
    plt.show()

if __name__ == '__main__':
    train_acc_compared()
    val_acc_compared()
    train_loss_compared()
    val_loss_compared()