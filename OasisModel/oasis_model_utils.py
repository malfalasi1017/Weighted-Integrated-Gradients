import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

class OasisModel(nn.Module):
    def __init__(self, num_classes=4):
        super(OasisModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def load_oasis_model(model_path='./OasisModel/best_oasis_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


    checkpoint = torch.load(model_path, map_location=device)
    label_names = checkpoint['label_names']

    model = OasisModel(num_classes=len(label_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, label_names, device

def load_labels_vocabulary(label_names):
    """Return the label names - equivalent to load_labels_vocabulary in inception_utils"""
    return label_names


def make_predictions_and_gradients(model, device):
    def predictions_and_gradients(images, target_label_index):
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        batch_predictions = []
        batch_gradients = []
        
        for img in images:
            if isinstance(img, np.ndarray):
                img_pil = Image.fromarray(img.astype(np.uint8))
            else:
                img_pil = img

            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            img_tensor.requires_grad_(True)
            

            output = model(img_tensor)
            predictions = F.softmax(output, dim=1)
            
            target_score = output[0, target_label_index]
            target_score.backward()
            
            gradients = img_tensor.grad.squeeze(0).cpu().numpy()
            gradients = np.transpose(gradients, (1, 2, 0))  
            
            batch_predictions.append(predictions.detach().cpu().numpy()[0])
            batch_gradients.append(gradients)
            
            img_tensor.grad = None
        
        return np.array(batch_predictions), np.array(batch_gradients)
    
    return predictions_and_gradients

def top_label_id_and_score(img, predictions_and_gradients_fn):
    """Get top prediction and score - equivalent to top_label_id_and_score"""
    predictions, _ = predictions_and_gradients_fn([img], 0)
    top_label_id = np.argmax(predictions[0])
    score = predictions[0][top_label_id]
    return top_label_id, score

def load_image(img_path):
    """Load image from path"""
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    return img_array