#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <caffe/caffe.hpp>

#include "json.hpp"
#include "predictor.hpp"
#include "timer.h"
#include "timer.impl.hpp"

using namespace caffe;
using std::string;
using json = nlohmann::json;

/* Pair (label, confidence) representing a prediction. */
using Prediction = std::pair<int, float>;

template <typename Dtype>
class StartProfile : public Net<Dtype>::Callback {
 public:
  explicit StartProfile(profile* prof, const shared_ptr<Net<Dtype>>& net)
      : prof_(prof), net_(net) {}

 protected:
  virtual void run(int layer) {
    auto e = new profile_entry(net_->layer_names()[layer].c_str(),
                               net_->layers()[layer]->type());
    prof_->add(layer, e);
  }

 private:
  profile* prof_;
  const shared_ptr<Net<Dtype>> net_;
};

template <typename Dtype>
class EndProfile : public Net<Dtype>::Callback {
 public:
  explicit EndProfile(profile* prof) : prof_(prof) {}

 protected:
  virtual void run(int layer) {
    auto e = prof_->get(layer);
    e->end();
  }

 private:
  profile* prof_;
};

class Predictor {
 public:
  Predictor(const string& model_file, const string& trained_file, int batch);

  std::vector<Prediction> Predict(float* imageData);

  shared_ptr<Net<float>> net_;
  int width_, height_, channels_;
  int batch_;
  profile* prof_{nullptr};
};

Predictor::Predictor(const string& model_file, const string& trained_file,
                     int batch) {
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  const auto input_layer = net_->input_blobs()[0];

  width_ = input_layer->width();
  height_ = input_layer->height();
  channels_ = input_layer->channels();
  batch_ = batch;

  CHECK(channels_ == 3 || channels_ == 1)
      << "Input layer should have 1 or 3 channels.";

  input_layer->Reshape(batch_, channels_, height_, width_);
  net_->Reshape();
}

/* Return the top N predictions. */
std::vector<Prediction> Predictor::Predict(float* imageData) {
  auto blob = new caffe::Blob<float>(batch_, channels_, height_, width_);
  blob->set_cpu_data(imageData);

  const std::vector<caffe::Blob<float>*> bottom{blob};

  if (prof_ != nullptr) {
    net_->add_before_forward(new StartProfile<float>(prof_, net_));
    net_->add_after_forward(new EndProfile<float>(prof_));
  }

  const auto rr = net_->Forward(bottom);
  const auto output_layer = rr[0];

  const auto len = output_layer->channels();
  const auto outputSize = len * batch_;
  const float* outputData = output_layer->cpu_data();

  std::vector<Prediction> predictions;
  predictions.reserve(outputSize);
  for (int cnt = 0; cnt < batch_; cnt++) {
    for (int idx = 0; idx < len; idx++) {
      predictions.emplace_back(
          std::make_pair(idx, outputData[cnt * len + idx]));
    }
  }

  return predictions;
}

PredictorContext New(char* model_file, char* trained_file, int batch) {
  try {
    const auto ctx = new Predictor(model_file, trained_file, batch);
    return (void*)ctx;
  } catch (const std::invalid_argument& ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

void Init() { ::google::InitGoogleLogging("go-caffe"); }

void StartProfiling(PredictorContext pred, const char* name,
                    const char* metadata) {
  auto predictor = (Predictor*)pred;
  if (name == nullptr) {
    name = "";
  }
  if (metadata == nullptr) {
    metadata = "";
  }
  predictor->prof_ = new profile(name, metadata);
}

void EndProfiling(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void DisableProfiling(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  if (predictor->prof_) {
    predictor->prof_->reset();
    delete predictor->prof_;
  }
  predictor->prof_ = nullptr;
}

char* ReadProfile(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  const auto s = predictor->prof_->read();
  const auto cstr = s.c_str();
  return strdup(cstr);
}

const char* Predict(PredictorContext pred, float* imageData) {
  auto predictor = (Predictor*)pred;
  const auto predictionsTuples = predictor->Predict(imageData);

  json predictions = json::array();
  for (const auto prediction : predictionsTuples) {
    predictions.push_back(
        {{"index", prediction.first}, {"probability", prediction.second}});
  }
  auto res = strdup(predictions.dump().c_str());

  return res;
}

int PredictorGetChannels(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  return predictor->channels_;
}

int PredictorGetWidth(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  return predictor->width_;
}

int PredictorGetHeight(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  return predictor->height_;
}

int PredictorGetBatchSize(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  return predictor->batch_;
}

void Delete(PredictorContext pred) {
  auto predictor = (Predictor*)pred;
  if (predictor->prof_) {
    predictor->prof_->reset();
    delete predictor->prof_;
    predictor->prof_ = nullptr;
  }
  delete predictor;
}

void SetMode(int mode) { Caffe::set_mode((caffe::Caffe::Brew)mode); }
