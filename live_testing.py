import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18

NUM_CLASSES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load("model_resnet18.pt", map_location=device, weights_only=True))
model.eval()


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Efteling PET Detector Active... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probabilities, 1)

    label = "PET" if pred.item() == 1 else "OTHER"
    color = (0, 255, 0) if label == "PET" else (0, 0, 255) # Green for PET, Red for Other
    score = conf.item() * 100

    cv2.putText(frame, f"{label} ({score:.1f}%)", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Efteling Live Waste Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()