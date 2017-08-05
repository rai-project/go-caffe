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

  input_layer->Reshape(1, channel_, height_, width_);
  net_->Reshape();

  // printf("width = %d\n", width_);
  // printf("height_ = %d\n", height_);
  // printf("channel_ = %d\n", channel_);

  CHECK(channel_ == 3 || channel_ == 1)
      << "Input layer should have 1 or 3 channels.";
}

/* Return the top N predictions. */
std::vector<Prediction> Predictor::Predict(float* imageData) {
  // for (int ii = 0; ii < width_ * height_ * channel_; ii++) {
  //   std::cout << imageData[ii] << "   ";
  // }
  // std::cout << std::endl;

  // Blob<float>* input_layer = net_->input_blobs()[0];
  // std::cout << "input_layer width = " << input_layer->width() << std::endl;
  // std::cout << "input_layer height = " << input_layer->height() << std::endl;
  // std::cout << "input_layer channels = " << input_layer->channels()
  //           << std::endl;
  // std::cout << "input shape = " << input_layer->shape()[3] << " "
  //           << input_layer->shape()[2] << " " << input_layer->shape()[1] << "
  //           "
  //           << std::endl;
  caffe::Blob<float>* blob =
      new caffe::Blob<float>(1, channel_, height_, width_);

  // std::cout << "blob shape = " << blob->shape()[3] << " " << blob->shape()[2]
  //           << " " << blob->shape()[1] << " " << std::endl;
  blob->set_cpu_data(imageData);
  std::vector<caffe::Blob<float>*> bottom{blob};

  const auto rr = net_->Forward(bottom);

  Blob<float>* output_layer = rr[0];

  const auto outputSize = output_layer->channels();
  const float* outputData = output_layer->cpu_data();

  std::vector<Prediction> predictions;
  predictions.reserve(outputSize);
  for (int idx = 0; idx < outputSize; idx++) {
    predictions.emplace_back(std::make_pair(idx, outputData[idx]));
  }

  // for (const auto pred : predictions) {
  //   std::cout << pred.first << "  " << pred.second << std::endl;
  // }

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
