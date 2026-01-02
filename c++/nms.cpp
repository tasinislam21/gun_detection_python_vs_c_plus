#include <vector>
#include <iostream>
#include <algorithm>
#include <torch/torch.h>
#include <torch/script.h>

float iou(torch::Tensor box1, torch::Tensor box2) {
    float x1_min = (box1[0].item<float>() - box1[2].item<float>())  / 2;
    float y1_min = (box1[1].item<float>() - box1[3].item<float>()) / 2;
    float x1_max = (box1[0].item<float>() + box1[2].item<float>()) / 2;
    float y1_max = (box1[1].item<float>() + box1[3].item<float>()) / 2;

    float x2_min = (box2[0].item<float>() - box2[2].item<float>()) / 2;
    float y2_min = (box2[1].item<float>() - box2[3].item<float>()) / 2;
    float x2_max = (box2[0].item<float>() + box2[2].item<float>()) / 2;
    float y2_max = (box2[1].item<float>() + box2[3].item<float>()) / 2;

    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);

    float inter_width = std::max(0.0f, inter_x_max - inter_x_min);
    float inter_height = std::max(0.0f, inter_y_max - inter_y_min);
    float inter_area = inter_width * inter_height;

    float area1 = box1[2].item<float>() * box1[3].item<float>();
    float area2 = box2[2].item<float>() * box2[3].item<float>();

    float union_area = area1 + area2 - inter_area;
    return inter_area / union_area;
}

std::vector<int> non_max_suppression(const torch::Tensor& result, float iou_threshold){
    std::vector<int> indices(result.size(1));
    std::vector<int> keep;

    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    torch::Tensor boxes = result.index({torch::indexing::Slice(0,4), torch::indexing::Slice()});
    torch::Tensor scores = result.index({5, torch::indexing::Slice()});
    
    std::vector<float> score_values(scores.size(0));
    
    for (size_t i = 0; i < scores.size(0); i++) {
        score_values[i] = scores[i].item<float>();
    }

    std::sort(indices.begin(), indices.end(),
            [&](int a, int b) {
                return score_values[a] > score_values[b];
            });

    while (!indices.empty()) {
        int current = indices.front();
        indices.erase(indices.begin()); 
        keep.push_back(current);

        std::vector<int> remaining;
        for (int i : indices) {
            if (iou(boxes.index({torch::indexing::Slice(0,4), current}), boxes.index({torch::indexing::Slice(0,4), i})) < iou_threshold) {
                remaining.push_back(i);
            }   
        }
        indices = std::move(remaining);
    }
    return keep;
}
