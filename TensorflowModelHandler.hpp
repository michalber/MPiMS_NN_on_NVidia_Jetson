/**
 * @file TensorflowModelHandler.hpp
 * @author Michał Berdzik
 * @brief 
 * @version 0.1
 * @date 2021-05-14
 * 
 * 
 */

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor_slice.h"

using tensorflow::ClientSession;
using tensorflow::int32;
using tensorflow::RunOptions;
using tensorflow::SavedModelBundle;
using tensorflow::Scope;
using tensorflow::SessionOptions;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::tstring;

namespace nn
{
    /**
     * @brief Function to decode image from file to Tensorflow tensors
     * 
     * @param filename path to file
     * @param out_tensors pointer to vector of tensors
     * @return Status 
     */
    Status ReadImageFile(const std::string &filename, std::vector<Tensor> *out_tensors);

    /**
     * @brief Struct to represent model prediction
     * 
     */
    struct Prediction
    {
        std::unique_ptr<std::vector<std::vector<float>>> boxes_;
        std::unique_ptr<std::vector<float>> scores_;
        std::unique_ptr<std::vector<int>> labels_;
    };

    /**
     * @brief Class to handle neural network prediction using Tensorflow API
     * 
     */
    class TensorflowModelHandler
    {
    public:
        /**
         * @brief Construct a new Tensorflow Model Handler object
         * 
         */
        TensorflowModelHandler(std::string filename);

        /**
         * @brief Destroy the Tensorflow Model Handler object
         * 
         */
        ~TensorflowModelHandler() = default;

        /**
         * @brief Run prediction on given image
         * 
         * @param filename path to image
         * @param out_pred Output preditions
         */
        void Predict(std::string filename, Prediction &out_pred);

    private:
        /**
         * @brief Function to run neural network interference
         * 
         * @param image_output Input tensors from image
         * @param pred Output preditions
         */
        void MakePrediction(std::vector<Tensor> &image_output, Prediction &pred);

    private:
        SavedModelBundle bundle_;
        SessionOptions session_options_;
        RunOptions run_options_;
    };
}
