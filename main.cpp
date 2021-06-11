/**
 * @file main.cpp
 * @author Michał Berdzik
 * @brief 
 * @version 0.1
 * @date 2021-05-14
 * 
 * 
 */

#include "./TensorflowModelHandler.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

constexpr float kModelThreshold = 0.8f;

void SaveImageWithPredictions(const std::string &in_path, const std::string &out_path, nn::Predictions &output_predictions)
{
    // Write prediction boxes to new image
    cv::Mat image_with_predictions = cv::imread(in_path, IMREAD_COLOR);

    Size size = image_with_predictions.size();
    const int height = size.height;
    const int width = size.width;

    auto boxes = (*output_predictions.boxes_);
    auto scores = (*output_predictions.scores_);

    for (int i = 0; i < boxes.size(); i++)
    {
        auto box = boxes[i];
        auto score = scores[i];
        if (score < kModelThreshold)
        {
            continue;
        }
        int ymin = (int)(box[0] * height);
        int xmin = (int)(box[1] * width);
        int h = (int)(box[2] * height) - ymin;
        int w = (int)(box[3] * width) - xmin;
        cv::Rect rect = cv::Rect(xmin, ymin, w, h);
        rectangle(image_with_predictions, rect, cv::Scalar(0, 0, 255), 2);
    }

    if (image_with_predictions.empty())
    {
        std::cout << " Failed to read image" << std::endl;
        return;
    }

    imwrite(out_path, image_with_predictions);
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cout << "Error! Usage: <path/to_saved_model> <path/to_input/image.jpg> <path/to/output/image.jpg>" << std::endl;
        return 1;
    }

    // Make a Prediction instance
    nn::Prediction output_predictions;
    output_predictions.boxes_ = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
    output_predictions.scores_ = std::unique_ptr<std::vector<float>>(new std::vector<float>());
    output_predictions.labels_ = std::unique_ptr<std::vector<int>>(new std::vector<int>());

    const std::string model_path = argv[1];
    const std::string test_image_file = argv[2];
    const std::string test_prediction_image = argv[3];

    // Load the saved_model
    nn::TensorflowModelHandler model_instance(model_path);

    //Predict on the input image
    model_instance.Predict(test_image_file, output_predictions);

    SaveImageWithPredictions(test_image_file, test_prediction_image, output_predictions);

    return 0;
}
