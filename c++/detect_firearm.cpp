#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

std::vector<int> non_max_suppression(const torch::Tensor& result, float iou_threshold);

struct PreprocessImage {
    torch::Tensor tensor;
    cv::Mat rgb_image;
};
    
torch::jit::script::Module model;

PreprocessImage preprocess_image(const cv::Mat& image_ori) {
    PreprocessImage out;
    cv::Mat resized, image_rgb;
    cv::resize(image_ori, resized, cv::Size(640, 640));
    cv::cvtColor(resized, image_rgb, cv::COLOR_BGR2RGB);
    // Convert to float tensor
    torch::Tensor img_tensor = torch::from_blob(
        image_rgb.data, {640, 640, 3}, torch::kUInt8);
    img_tensor = img_tensor.to(torch::kFloat32);
    // HWC → CHW
    img_tensor = img_tensor.permute({2, 0, 1});
    // Normalize
    img_tensor = img_tensor / 255.0;
    // Add batch dimension
    img_tensor = img_tensor.unsqueeze(0);
    out.tensor = img_tensor;
    out.rgb_image = image_rgb.clone();
    return out;
}

void load_model(torch::Device device) {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load("../../best.torchscript", device=device);
    std::cout << "Model loaded successfully\n";
}

void filter_result(torch::Tensor& result) {
    torch::Tensor scores = result.index({5, torch::indexing::Slice()});
    torch::Tensor mask = scores > 0.35;
    result = result.index({torch::indexing::Slice(), mask});
}

void draw_bounding_box(PreprocessImage& preprocessed_images, torch::Tensor& result, std::vector<int>& indexes) {
    // Drawing bounding boxes on the frame 
    for (int index: indexes) {
        torch::Tensor box_tensor = result.index({torch::indexing::Slice(0, 4), index});
        box_tensor = box_tensor.cpu();
        auto box_np = box_tensor.contiguous().data_ptr<float>();
        float x = box_np[0];
        float y = box_np[1];
        float w = box_np[2];
        float h = box_np[3];
        int x1 = int(x - w / 2);
        int y1 = int(y - h / 2);
        int x2 = int(x1 + w);
        int y2 = int(y1 + h);
        cv::rectangle(preprocessed_images.rgb_image, cv::Point(x1, y1), cv::Point(x2, y2),
                        cv::Scalar(0, 0, 255), 2);
    }
}

void include_inference_time(std::chrono::milliseconds& duration, PreprocessImage& preprocessed_images) {
    std::string text = "Inference: " + std::to_string(duration.count()) + " ms"; 
    int font = cv::FONT_HERSHEY_SIMPLEX; 
    double fontScale = 0.7; int thickness = 2; 
    cv::Point org(10, 30); // top-left corner 
    cv::Scalar color(0, 255, 0); // green 
    cv::putText(preprocessed_images.rgb_image, text, org, font, fontScale, color, thickness);

}

int main() {
    torch::Device device(torch::kCUDA);
    load_model(device);
    
    cv::VideoCapture cap("../../evaluation.mp4");
    cv::Mat frame;
    
    torch::NoGradGuard no_grad;
    while (cap.read(frame)) {
        PreprocessImage preprocessed_images = preprocess_image(frame);
        torch::Tensor input = preprocessed_images.tensor;
        input = input.to(device);
        auto start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor result = model.forward({input}).toTensor()[0]; //extract the first element from the batch
        auto end_time = std::chrono::high_resolution_clock::now();
        filter_result(result);  // we only want confidence scores greater than 35%
        std::vector<int> supressed_indexes;
        if (result.defined() && result.numel() != 0 && result.dim() == 2 && result.size(1) > 0) {
            supressed_indexes = non_max_suppression(result, 0.5f);
        }
        draw_bounding_box(preprocessed_images, result, supressed_indexes);
        std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        include_inference_time(duration, preprocessed_images);
        cv::imshow("Frame", preprocessed_images.rgb_image);
            if ((cv::waitKey(25) & 0xFF) == 'q'){
                break;
                }
        }
    return 0;
}
