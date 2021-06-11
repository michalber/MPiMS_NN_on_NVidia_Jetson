/**
 * @file TensorflowModelHandler.cpp
 * @author Michał Berdzik
 * @brief 
 * @version 0.1
 * @date 2021-05-14
 * 
 * 
 */

#include "TensorflowModelHandler.hpp"

namespace nn
{
    Status ReadImageFile(const std::string &filename, std::vector<Tensor> *out_tensors)
    {
        using namespace ::tensorflow::ops;
        Scope root = Scope::NewRootScope();
        auto output = tensorflow::ops::ReadFile(root.WithOpName("file_reader"), filename);

        tensorflow::Output image_reader;
        const int wanted_channels = 3;
        image_reader = tensorflow::ops::DecodeJpeg(root.WithOpName("file_decoder"), output, DecodeJpeg::Channels(wanted_channels));

        auto image_unit8 = Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);
        auto image_expanded = ExpandDims(root.WithOpName("expand_dims"), image_unit8, 0);

        tensorflow::GraphDef graph;
        auto s = (root.ToGraphDef(&graph));

        if (!s.ok())
        {
            std::cout << "Error in loading image from file" << std::endl;
        }
        else
        {
            std::cout << "Loaded correctly" << std::endl;
        }

        ClientSession session(root);

        auto run_status = session.Run({image_expanded}, out_tensors);
        if (!run_status.ok())
        {
            std::cout << "Error in running session" << std::endl;
        }
        return Status::OK();
    }

    TensorflowModelHandler::TensorflowModelHandler(std::string path)
    {
        auto status = tensorflow::LoadSavedModel(session_options_, run_options_, path, {"serve"},
                                                 &bundle_);

        if (status.ok())
        {
            std::cout << "Model loaded successfully" << std::endl;
        }
        else
        {
            std::cout << "Error in loading model" << std::endl;
        }
    }

    void TensorflowModelHandler::Predict(std::string filename, Prediction &out_pred)
    {
        std::vector<Tensor> image_output;
        auto read_status = ReadImageFile(filename, &image_output);
        MakePrediction(image_output, out_pred);
    }

    void TensorflowModelHandler::MakePrediction(std::vector<Tensor> &image_output, Prediction &out_pred)
    {
        const std::string input_node = "serving_default_input_tensor:0";
        std::vector<std::pair<std::string, Tensor>> inputs_data = {{input_node, image_output[0]}};
        std::vector<std::string> output_nodes = {{"StatefulPartitionedCall:0",   //detection_anchor_indices
                                                  "StatefulPartitionedCall:1",   //detection_boxes
                                                  "StatefulPartitionedCall:2",   //detection_classes
                                                  "StatefulPartitionedCall:3",   //detection_multiclass_scores
                                                  "StatefulPartitionedCall:4",   //detection_scores
                                                  "StatefulPartitionedCall:5"}}; //num_detections

        std::vector<Tensor> predictions;
        this->bundle_.GetSession()->Run(inputs_data, output_nodes, {}, &predictions);

        auto predicted_boxes = predictions[1].tensor<float, 3>();
        auto predicted_scores = predictions[4].tensor<float, 2>();
        auto predicted_labels = predictions[2].tensor<float, 2>();

        for (int i = 0; i < 100; i++)
        {
            std::vector<float> coords;
            for (int j = 0; j < 4; j++)
            {
                coords.push_back(predicted_boxes(0, i, j));
            }
            (*out_pred.boxes_).push_back(coords);
            (*out_pred.scores_).push_back(predicted_scores(0, i));
            (*out_pred.labels_).push_back(predicted_labels(0, i));
        }
    }
} // namespace nn
