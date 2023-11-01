import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class Encoder(nn.Module):
    def __init__(self, num_features=256):
        super(Encoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_block1(x) # 112 * 112
        x = self.conv_block2(x) # 56 * 56
        x = self.conv_block3(x) # 28 * 28

        return x

class AccidentDetector(nn.Module):
    def __init__(self, num_features=256, num_classes=2, num_layers=3, num_heads=8, dropout=0.3):
        super(AccidentDetector, self).__init__()
        self.num_features = num_features

        self.encoder = Encoder(num_features=num_features)

        self.reduce_dim = nn.Linear(num_features * 28 * 28, 256)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=num_features, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

        self.fc = nn.Linear(num_features, num_classes)


    def forward(self, x):

        # 输入尺寸: (batch_size, seq_length, 3, H, W)
        batch_size, seq_length = x.size(0), x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))  # 将 batch_size 和 seq_length 合并为一个维度

        features = self.encoder(x)
        features = features.view(batch_size, seq_length, self.num_features * 28 * 28)  # 调整尺寸以适应 Transformer 输入

        features = self.reduce_dim(features)
        features = features.permute(1, 0, 2)  # 调整尺寸以适应 Transformer 输入

        memory = torch.zeros_like(features)
        output = self.decoder(features, memory)
        output = output.permute(1, 0, 2)  # 调整尺寸以适应全连接层输入

        output = self.fc(output)

        return output

class DCNN(nn.Module):
    def __init__(self, num_features=256, num_classes=2, dropout_rate=0.3):
        super(DCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(num_features * 56 * 56, num_classes)

    def forward(self, x):

        x = x.squeeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

def preprocess(image: Image.Image):
    preprocess_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the input size that your model expects
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess_pipeline(image).unsqueeze(0)  # Add batch dimension

def detect_accident(model, video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    accident_detected = False

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            prediction = model(image_tensor)

        # Assuming that the accident class is now the first class
        if prediction.argmax().item() == 0:
            accident_detected = True
            break

    cap.release()

    return accident_detected

if __name__ == '__main__':

    model = DCNN()
    state_dict = torch.load('././model/DCNN.pkl')

    # 去除nn.DataParallel带来的module前缀
    new_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd)
    model.eval()

    video_path = '././dataset/bboxes_video/000007.mp4'
    if detect_accident(model, video_path):
        print("Accident Detected")
    else:
        print("No Accident Detected")