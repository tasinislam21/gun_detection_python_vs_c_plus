#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

struct PreprocessResult {
    torch::Tensor tensor;
    cv::Mat rgb_image;
};

PreprocessResult preprocess_image(const cv::Mat& image_ori) {
    PreprocessResult out;
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

int main() {
    torch::Device device(torch::kCUDA);
    //torch::Device device(torch::kCPU);
    torch::jit::script::Module model = torch::jit::load("../../best.torchscript", device=device);   
    std::cout << "Created model success!" << std::endl;
    cv::VideoCapture cap("../../evaluation.mp4");
    cv::Mat frame;
    
    torch::NoGradGuard no_grad;
    while (cap.read(frame)) {
        PreprocessResult preprocessed_images = preprocess_image(frame);
        torch::Tensor input = preprocessed_images.tensor;
        input = input.to(device);
        auto start_time = std::chrono::high_resolution_clock::now();
        auto results = model.forward({input}).toTensor();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto result = results[0];
        
        auto boxes = result.slice(0, 0, 4);  // column 0..3
        auto person_prob = std::get<1>(result[4].max(0));
        auto gun_prob = std::get<1>(result[5].max(0));

        float gun_conf = result[5][gun_prob].item<float>();

        if (gun_conf > 0.35) {
                torch::Tensor box_tensor = boxes.index({torch::indexing::Slice(), gun_prob});
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
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::string text = "Inference: " + std::to_string(duration.count()) + " ms"; 
        int font = cv::FONT_HERSHEY_SIMPLEX; 
        double fontScale = 0.7; int thickness = 2; 
        cv::Point org(10, 30); // top-left corner 
        cv::Scalar color(0, 255, 0); // green 
        cv::putText(preprocessed_images.rgb_image, text, org, font, fontScale, color, thickness);
        cv::imshow("Frame", preprocessed_images.rgb_image);
            if ((cv::waitKey(25) & 0xFF) == 'q'){
                break;
                }
        }
    return 0;
    }