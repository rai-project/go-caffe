#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe/caffe.hpp>

#include "json.hpp"
#include "predictor.hpp"

using namespace caffe;
using std::string;
using json = nlohmann::json;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

class Predictor {
 public:
  Predictor(const string& model_file, const string& trained_file);

  std::vector<Prediction> Predict(const char* imageData, int imageRows,
                                  int imageCols, int imageChannels);

 private:
  std::vector<float> iPredict(const char* imageData, int imageRows,
                              int imageCols, int imageChannels);

 private:
  shared_ptr<Net<float> > net_;
  int width_, height_, channel_;
};

Predictor::Predictor(const string& model_file, const string& trained_file) {
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];

  width_ = input_layer->width();
  height_ = input_layer->height();
  channel_ = input_layer->channels();

  CHECK(channel_ == 3 || channel_ == 1)
      << "Input layer should have 1 or 3 channels.";
}

std::vector<float> Predictor::iPredict(const char* imageData, int imageRows,
                                       int imageCols, int imageChannels) {

  const auto imageSize = imageRows * imageCols * imageChannels;
  std::vector<float> data;
  data.reserve(imageSize);
  std::transform(imageData, imageData + imageSize, data.begin() ,
                 [](const char pixel) -> float { return pixel / 255.0f; });

  caffe::Blob<float> blob{1, channel_, height_, width_};
  blob.set_cpu_data(data.data());
  std::vector<caffe::Blob<float>*> bottom;
  bottom.push_back(&blob);

  net_->Forward(bottom);

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Return the top N predictions. */
std::vector<Prediction> Predictor::Predict(const char* imageData, int imageRows,
                                           int imageCols, int imageChannels) {
  std::vector<float> output =
      iPredict(imageData, imageRows, imageCols, imageChannels);
  const auto outputSize = output.size();

  std::vector<Prediction> predictions;
  predictions.reserve(outputSize);
  for (int idx = 0; idx < outputSize; idx++) {
    predictions.emplace_back(std::make_pair(idx, output[idx]));
  }

  return predictions;
}

PredictorContext New(char* model_file, char* trained_file) {
  try {
    ::google::InitGoogleLogging("inference_server");

    const auto ctx = new Predictor(model_file, trained_file);
    return (void*)ctx;
  } catch (const std::invalid_argument& ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

const char* Predict(PredictorContext pred, const char* imageData,
                    int imageRows, int imageCols, int imageChannels) {
  auto predictor = (Predictor*)pred;
  auto predictions =
      predictor->Predict(imageData, imageRows, imageCols, imageChannels);

  json j;
  j["predictions"] = predictions;
  auto res = strdup(j.dump().c_str());
  return res;
}

void Delete(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  delete predictor;
}

void SetMode(int mode) { Caffe::set_mode((caffe::Caffe::Brew) mode); }
