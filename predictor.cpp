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

  std::vector<Prediction> Predict(float* imageData);

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

  printf("width = %d\n", width_);
  printf("height_ = %d\n", height_);
  printf("channel_ = %d\n", channel_);

  CHECK(channel_ == 3 || channel_ == 1)
      << "Input layer should have 1 or 3 channels.";
}

/* Return the top N predictions. */
std::vector<Prediction> Predictor::Predict(float* imageData) {
  net_->Forward(imageData);

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  const auto output = std::vector<float>(begin, end);

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
    const auto ctx = new Predictor(model_file, trained_file);
    return (void*)ctx;
  } catch (const std::invalid_argument& ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

void Init() { ::google::InitGoogleLogging("inference_server"); }

const char* Predict(PredictorContext pred, float* imageData) {
  auto predictor = (Predictor*)pred;
  auto predictionsTuples = predictor->Predict(imageData);

  json predictions = json::array();
  for (const auto prediction : predictionsTuples) {
    predictions.push_back(
        {{"index", prediction.first}, {"probability", prediction.second}});
  }
  auto res = strdup(predictions.dump().c_str());
  return res;
}

void Delete(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  delete predictor;
}

void SetMode(int mode) { Caffe::set_mode((caffe::Caffe::Brew)mode); }
