import cv2
import torch
import time
import PostProcess

device = 'cuda'
#device = 'cpu'

torchmodel = torch.jit.load("../best.torchscript", map_location=device)
torchmodel.eval()

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


if __name__ == '__main__':
    cap = cv2.VideoCapture('../evaluation.mp4')
    postprocessor = PostProcess.PostProcessor()
    with torch.no_grad():
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                image, image_rgb = preprocess_image(frame)
                image = image.to(device)
                result, duration_ms = run_model()
                postprocessor.set_image(image_rgb)
                postprocessor.set_time(duration_ms)
                postprocessor.set_result(result)
                cv2.imshow('Frame', postprocessor.get_frame())

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

    cap.release()
    cv2.destroyAllWindows()
