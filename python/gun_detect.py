import cv2
import torch
import time

def preprocess_image(image_ori) -> torch.Tensor:
    image = cv2.resize(image_ori, (640, 640))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
    image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, image_rgb

device = 'cuda'
#device = 'cpu'

torchmodel = torch.jit.load("../best.torchscript", map_location=device)
torchmodel.eval()
cap = cv2.VideoCapture('../evaluation.mp4')

with torch.no_grad():
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            image, image_rgb = preprocess_image(frame)
            image = image.to(device)
            start_time = time.time() 
            results = torchmodel(image)
            print(results.shape)
            end_time = time.time()
            result = results[0]  # Remove the batch dimension if there's only one image
            boxes = result[:4, :]
            person_prob = torch.argmax(result[4])
            gun_prob = torch.argmax(result[5])
            if result[5][gun_prob] > 0.35:
                box_np = boxes[:, gun_prob].cpu().numpy() if boxes[:, gun_prob].is_cuda else boxes[:, gun_prob].numpy()
                x, y, w, h = box_np
                x1 = int((x - w / 2))
                y1 = int((y - h / 2))
                x2 = int(x1 + w)
                y2 = int(y1 + h)
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
            duration_ms = (end_time - start_time) * 1000
            text = f"Inference: {duration_ms:.1f} ms"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            org = (10, 30) # top-left corner
            color = (0, 255, 0) # green (BGR)
            cv2.putText(image_rgb, text, org, font, font_scale, color, thickness)
            cv2.imshow('Frame', image_rgb)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

cap.release()
cv2.destroyAllWindows()
