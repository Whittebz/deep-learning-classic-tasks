import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.v2 as T
from PIL import Image

class ObjectDetector:
    def __init__(self, model_path='models/faster_rcnn_pennfudan.pth'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # 2 classes: background and person
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model weights. Error: {e}")
            self.model = None

        self.transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])

    def predict(self, image_input):
        if self.model is None:
            return None, "Error: Model weights not found. Train the model first."
            
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
            
        img_tensor = self.transform(image).to(self.device)
        
        with torch.no_grad():
            prediction = self.model([img_tensor])[0]
            
        # Filter predictions by score > 0.5
        boxes = prediction['boxes']
        scores = prediction['scores']
        labels = prediction['labels']
        
        keep = scores > 0.5
        boxes = boxes[keep]
        scores = scores[keep]
        
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from io import BytesIO
        
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, f'Person {score:.2f}', color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            
        plt.axis('off')
        
        # Save plot to PIL Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        result_img = Image.open(buf)
        plt.close(fig)
        
        return result_img, f"Found {len(boxes)} persons."

if __name__ == '__main__':
    detector = ObjectDetector()
