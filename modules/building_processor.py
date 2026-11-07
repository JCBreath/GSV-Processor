from ultralytics import YOLO
import supervision as sv
from PIL import Image
import cv2
import torch
from torchvision import transforms
import numpy as np

class BuildingProcessor:
    def __init__(self, model_paths={
            'building_detect': 'ckpt/building_crop_yolo2500_idaho829.pt',
            'wallmat': 'ckpt/wallmat_efficientnet_s_384_2w+isvc24.pth',
            'condition': 'ckpt/condition_efficientnet_m_384_isvc25.pth'
        }):
        building_detect_model_path = model_paths.get('building_detect', None)
        if building_detect_model_path is not None:
            print("Loading building detection model...")
            self.building_detect_model = YOLO(building_detect_model_path).to('cuda')
        else:
            self.building_detect_model = None



        wallmat_model_path = model_paths.get('wallmat', None)
        if wallmat_model_path is not None:
            print("Loading wall material classification model...")
            self.wallmat_model = torch.load(wallmat_model_path, map_location='cuda', weights_only=False)
            self.wallmat_model.eval()
            input_size = 384
            self.wallmat_config = {
                "input_size": input_size,
                "classes": ['brick', 'other', 'wood', 'wood & brick'],
                "image_processor": transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }
        else:
            self.wallmat_model = None

        condition_model_path = model_paths.get('condition', None)
        if condition_model_path is not None:
            print("Loading building condition classification model...")
            import torch.nn as nn
            import torchvision.models as models
            model = models.efficientnet_v2_m(pretrained=True) # Default EfficientNet_B0_Weights.IMAGENET1K_V1
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier[1].in_features, 1),
                nn.Sigmoid()  # ensures output is in [0, 1]
            )
            model.load_state_dict(torch.load(condition_model_path, map_location='cuda'))
            model.eval()
            self.condition_model = model.to('cuda')
            # self.condition_model = torch.load(condition_model_path, map_location='cuda', weights_only=False)
            self.condition_model.eval()
            self.condition_config = {
                "input_size": input_size,
                "image_processor": transforms.Compose([
                    transforms.Resize((480, 480)),
                    transforms.CenterCrop(480),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }
            
        else:
            self.condition_model = None

        self.wallmat_vqa_model = None
        self.wallmat_vqa_vis_processors = None
        self.wallmat_vqa_txt_processors = None

    def detect_and_crop(self, image: Image.Image):
        if self.building_detect_model is None:
            raise ValueError("Building detection model is not loaded.")

        image = np.array(image)
        height, width, channels = image.shape

        results = self.building_detect_model.predict(source=cv2.cvtColor(image, cv2.COLOR_RGB2BGR), conf=0.25, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        if len(detections.class_id) == 0:
            return None

        for i, (xyxy, _, confidence, class_id, _, _) in enumerate(detections):
            xyxy[0] = max(xyxy[0], 0.0)
            xyxy[1] = max(xyxy[1], 0.0)
            xyxy[2] = min(xyxy[2], width - 1)
            xyxy[3] = min(xyxy[3], height - 1)

            # Not the target
            if xyxy[0] > (width / 2 + 10) or xyxy[2] < (width / 2 - 10):
                continue
            
            return {
                'result': Image.fromarray(image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]),
                'metadata': {
                    'xyxy': xyxy,
                    'confidence': confidence,
                    'class_id': class_id,
                }
            }

        return None

    def get_wallmat(self, image: Image.Image):
        if self.wallmat_model is None:
            raise ValueError("Wall material classification model is not loaded.")
        
        image = self.wallmat_config["image_processor"](image).float().unsqueeze(0).to('cuda') 
        output = self.wallmat_model(image)
        _, pred = torch.max(output,1)

        return {
            'material': self.wallmat_config["classes"][pred],
            'output': output.detach().cpu().numpy().tolist()
        }

    def get_wallmat_vqa(self, image: Image.Image):
        if self.wallmat_vqa_model is None:
            print("Loading BLIP VQA wall material model...")
            try:
                from lavis.models import load_model_and_preprocess
            except ImportError as exc:
                raise ImportError(
                    "get_wallmat_vqa() requires the LAVIS package."
                ) from exc

            (
                self.wallmat_vqa_model,
                self.wallmat_vqa_vis_processors,
                self.wallmat_vqa_txt_processors,
            ) = load_model_and_preprocess(
                name="blip_vqa",
                model_type="vqav2",
                is_eval=True,
                device="cuda",
            )

        questions = {
            "building_material": "What material is this building made of",
            "is_wood_1": "Is this building made of wood",
            "is_wood_2": "Is this wall made of wood",
            "is_wood_3": "Does this building have any wood parts",
            "is_brick_1": "Is this building made of brick",
            "is_brick_2": "Is this wall made of brick",
            "is_brick_3": "Does this building have any brick parts",
            "is_brick_4": "Does this building have any masonry structure",
        }

        processed_image = self.wallmat_vqa_vis_processors["eval"](image).unsqueeze(0).to("cuda")
        answers = {}
        with torch.inference_mode():
            for key, question in questions.items():
                processed_question = self.wallmat_vqa_txt_processors["eval"](question)
                answers[key] = self.wallmat_vqa_model.predict_answers(
                    samples={
                        "image": processed_image,
                        "text_input": processed_question,
                    },
                    inference_method="generate",
                )[0]

        is_wood = any(answers[key] == "yes" for key in ("is_wood_1", "is_wood_2", "is_wood_3"))
        is_brick = any(
            answers[key] == "yes"
            for key in ("is_brick_1", "is_brick_2", "is_brick_3", "is_brick_4")
        )

        if answers["building_material"] in ("brick", "wood"):
            material = "wood & brick" if is_wood and is_brick else answers["building_material"]
        else:
            material = "other"

        return {
            "material": material,
            "answers": answers,
            "is_wood": is_wood,
            "is_brick": is_brick,
        }
    
    def get_condition(self, image: Image.Image):
        if self.condition_model is None:
            raise ValueError("Building condition classification model is not loaded.")
        
        image = self.condition_config["image_processor"](image).float().unsqueeze(0).to('cuda') 
        output = self.condition_model(image)

        return {
            'score': min(max(int(round(output.item() * 4 + 1)), 1), 5),
            'output': output.item()
        }

if __name__ == "__main__":
    pass
    
