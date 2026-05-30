import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.v2 as T
from PIL import Image, ImageDraw


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
            print('Model loaded successfully.')
        except Exception as e:
            print(f'Failed to load model weights. Error: {e}')
            self.model = None

        self.transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])

    def predict(self, image_input):
        if self.model is None:
            return None, 'Error: Model weights not found. Train the model first.'

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
        keep = scores > 0.5
        boxes = boxes[keep]
        scores = scores[keep]

        result_img = image.copy()
        draw = ImageDraw.Draw(result_img)
        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            draw.rectangle(
                [(float(xmin), float(ymin)), (float(xmax), float(ymax))],
                outline='red',
                width=3,
            )
            draw.text(
                (float(xmin) + 4, max(0.0, float(ymin) - 16)),
                f'Person {score:.2f}',
                fill='red',
            )

        return result_img, f'Found {len(boxes)} persons.'


if __name__ == '__main__':
    detector = ObjectDetector()
