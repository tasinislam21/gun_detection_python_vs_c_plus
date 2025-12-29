import cv2
import torch
import time
import nms

def preprocess_image(image_ori) -> torch.Tensor:
    image = cv2.resize(image_ori, (640, 640))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).float()
    image_tensor = image_tensor.permute(2, 0, 1)  # Change from HWC to CHW format
    image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, image_rgb

def run_model():
    start_time = time.time()
    result = torchmodel(image)[0]
    end_time = time.time()
    return result, (end_time - start_time) * 1000

def filter_result():
    boxes = result[:4, :]
    scores = result[5]

    mask = scores > 0.35    # boolean mask

    if mask.any():  # at least one score passes the threshold
        filtered_boxes = boxes[:, mask]
        filtered_scores = scores[mask]
        # Convert to list of tensors to mimic original behavior
        filtered_boxes = [filtered_boxes[:, i] for i in range(filtered_boxes.shape[1])]
        filtered_scores = filtered_scores.tolist()
    else:
        filtered_boxes = []
        filtered_scores = []

    return filtered_boxes, filtered_scores


def get_distinct_indices():
    return nms.non_max_suppression(filtered_boxes, filtered_scores, 0.5)

def draw_bbox():
    for box in nms_boxes:
        box_np = box.cpu().numpy()
        x, y, w, h = box_np
        x1 = int((x - w / 2))
        y1 = int((y - h / 2))
        x2 = int(x1 + w)
        y2 = int(y1 + h)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

def put_inference_time():
    text = f"Inference: {duration_ms:.1f} ms"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    org = (10, 30)  # top-left corner
    color = (0, 255, 0)  # green (BGR)
    cv2.putText(image_rgb, text, org, font, font_scale, color, thickness)
    cv2.imshow('Frame', image_rgb)

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
            result, duration_ms = run_model()
            filtered_boxes, filtered_scores = filter_result()
            distinct_indices = get_distinct_indices()
            if len(filtered_boxes) > 0:
                distinct_indices = nms.non_max_suppression(filtered_boxes, filtered_scores, iou_threshold=0.5)
                nms_boxes = [filtered_boxes[i] for i in distinct_indices]
            else:
                nms_boxes = []
            draw_bbox()
            put_inference_time()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

cap.release()
cv2.destroyAllWindows()
