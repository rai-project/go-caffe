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
  explicit StartProfile(profile *prof, const shared_ptr<Net<Dtype>> &net)
      : prof_(prof), net_(net) {}
  virtual ~StartProfile() {}

 protected:
  virtual void run(int layer) final {
    if (prof_ == nullptr || net_ == nullptr) {
      return;
    }
    const auto layer_name = net_->layer_names()[layer];
    const auto layer_type = net_->layers()[layer]->type();
    auto e = new profile_entry(layer_sequence_index_, layer_name, layer_type);
    prof_->add(layer, e);
    order_++;
  }

 private:
  profile *prof_{nullptr};
  int order_{1};
  const shared_ptr<Net<Dtype>> net_{nullptr};
};

template <typename Dtype>
class EndProfile : public Net<Dtype>::Callback {
 public:
  explicit EndProfile(profile *prof) : prof_(prof) {}
  virtual ~EndProfile() {}

 protected:
  virtual void run(int layer) final {
    if (prof_ == nullptr) {
      return;
    }
    auto e = prof_->get(layer);
    if (e == nullptr) {
      return;
    }
    e->end();
  }

 private:
  profile *prof_{nullptr};
};

#if 0
#define DEBUG_STMT std::cout << __func__ << "  " << __LINE__ << "\n";
#else
#define DEBUG_STMT
#endif

class Predictor {
 public:
  Predictor(const string &model_file, const string &trained_file, int batch,
            caffe::Caffe::Brew mode);

  std::vector<Prediction> Predict(float *imageData);

  void setMode() {
    Caffe::set_mode(mode_);
    if (mode_ == Caffe::Brew::GPU) {
      Caffe::SetDevice(0);
    }
  }

  shared_ptr<Net<float>> net_;
  int width_, height_, channels_;
  int batch_;
  caffe::Caffe::Brew mode_{Caffe::CPU};
  profile *prof_{nullptr};
  bool prof_registered_{false};
};

Predictor::Predictor(const string &model_file, const string &trained_file,
                     int batch, caffe::Caffe::Brew mode) {
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  mode_ = mode;

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
std::vector<Prediction> Predictor::Predict(float *imageData) {
  DEBUG_STMT

  setMode();
  DEBUG_STMT

  // auto mallocEntry = new profile_entry("create blob", "malloc");
  // prof_->add(24, mallocEntry);

  auto blob = new caffe::Blob<float>(batch_, channels_, height_, width_);

  // mallocEntry->end();

  // auto setDataEntry = new profile_entry("set data", "setData");
  // prof_->add(025, setDataEntry);
  DEBUG_STMT
  if (mode_ == Caffe::CPU) {
    DEBUG_STMT
    blob->set_cpu_data(imageData);
  } else {
    DEBUG_STMT
    blob->set_gpu_data(imageData);
    blob->mutable_gpu_data();
  }
  DEBUG_STMT
  // setDataEntry->end();

  const std::vector<caffe::Blob<float> *> bottom{blob};
  DEBUG_STMT
  StartProfile<float> *start_profile = nullptr;
  EndProfile<float> *end_profile = nullptr;
  if (prof_ != nullptr && prof_registered_ == false) {
    DEBUG_STMT
    start_profile = new StartProfile<float>(prof_, net_);
    end_profile = new EndProfile<float>(prof_);
    net_->add_before_forward(start_profile);
    net_->add_after_forward(end_profile);
    prof_registered_ = true;
  }
  DEBUG_STMT
  // net_->set_debug_info(true);

  // auto netForwardEntry = new profile_entry("net forward", "netForward");
  // prof_->add(1026, netForwardEntry);
  const auto rr = net_->Forward(bottom);
  const auto output_layer = rr[0];
  // netForwardEntry->end();

  // auto copyBackEntry = new profile_entry("copy back", "copyBack");
  // prof_->add(1027, copyBackEntry);
  const auto len = output_layer->channels();
  const auto outputSize = len * batch_;
  const float *outputData = output_layer->cpu_data();

  std::vector<Prediction> predictions;
  predictions.reserve(outputSize);
  for (int cnt = 0; cnt < batch_; cnt++) {
    for (int idx = 0; idx < len; idx++) {
      predictions.emplace_back(
          std::make_pair(idx, outputData[cnt * len + idx]));
    }
  }
  DEBUG_STMT
  // copyBackEntry->end();

  /*
  if (start_profile) {
    delete start_profile;
  }
  if (end_profile) {
    delete end_profile;
  }
  */

  return predictions;
}

PredictorContext CaffeNew(char *model_file, char *trained_file, int batch,
                          int mode) {
  try {
    const auto ctx = new Predictor(model_file, trained_file, batch,
                                   (caffe::Caffe::Brew)mode);
    return (void *)ctx;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

void CaffeInit() { ::google::InitGoogleLogging("go-caffe"); }

void CaffeStartProfiling(PredictorContext pred, const char *name,
                         const char *metadata) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (name == nullptr) {
    name = "";
  }
  if (metadata == nullptr) {
    metadata = "";
  }
  if (predictor->prof_ == nullptr) {
    predictor->prof_ = new profile(name, metadata);
  } else {
    predictor->prof_->reset();
  }
}

void CaffeEndProfiling(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void CaffeDisableProfiling(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
  }
}

char *CaffeReadProfile(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return strdup("");
  }
  if (predictor->prof_ == nullptr) {
    return strdup("");
  }
  const auto s = predictor->prof_->read();
  const auto cstr = s.c_str();
  return strdup(cstr);
}

const char *CaffePredict(PredictorContext pred, float *imageData) {
  DEBUG_STMT;

  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return strdup("");
  }
  DEBUG_STMT;

  const auto predictionsTuples = predictor->Predict(imageData);

  DEBUG_STMT;

  json predictions = json::array();
  for (const auto prediction : predictionsTuples) {
    predictions.push_back(
        {{"index", prediction.first}, {"probability", prediction.second}});
  }
  DEBUG_STMT;

  auto res = strdup(predictions.dump().c_str());

  DEBUG_STMT;

  return res;
}

int CaffePredictorGetChannels(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->channels_;
}

int CaffePredictorGetWidth(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->width_;
}

int CaffePredictorGetHeight(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->height_;
}

int CaffePredictorGetBatchSize(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->batch_;
}

void CaffeDelete(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
    delete predictor->prof_;
    predictor->prof_ = nullptr;
  }
  delete predictor;
}

void CaffeSetMode(int mode) {
  static bool mode_set = false;
  if (!mode_set) {
    mode_set = true;
    Caffe::set_mode((caffe::Caffe::Brew)mode);
    // Caffe::set_mode(Caffe::Brew::GPU);
    if (mode == Caffe::Brew::GPU) {
      Caffe::SetDevice(0);
    }
    std::cout << Caffe::mode() << "  mode = " << mode << "\n";
  }
}
