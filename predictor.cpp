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

#if 0
#define DEBUG_STMT std ::cout << __func__ << "  " << __LINE__ << "\n";
#else
#define DEBUG_STMT
#endif

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
    const auto blobs = net_->layers()[layer]->blobs();
    shapes_t shapes{};
    for (const auto blob : blobs) {
      shapes.emplace_back(blob->shape());
    }
    auto e = new profile_entry(current_layer_sequence_index_, layer_name,
                               layer_type, shapes);
    prof_->add(layer, e);
    current_layer_sequence_index_++;
  }

 private:
  profile *prof_{nullptr};
  int current_layer_sequence_index_{1};
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

class Predictor {
 public:
  Predictor(const string &model_file, const string &trained_file,
            int batch_size, caffe::Caffe::Brew mode);

  void Predict(float *inputData);

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
  int pred_len_;
  std::vector<float *> results_{nullptr};
  int *output_indexes_;
  profile *prof_{nullptr};
  bool profile_enabled_{false};
};

Predictor::Predictor(const string &model_file, const string &trained_file,
                     int batch_size, std::vector<int> output_indexes,
                     caffe::Caffe::Brew mode) {
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  mode_ = mode;
  output_indexes_ = output_indexes;

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  const auto input_layer = net_->input_blobs()[0];

  width_ = input_layer->width();
  height_ = input_layer->height();
  channels_ = input_layer->channels();
  batch_ = batch_size;

  CHECK(channels_ == 3 || channels_ == 1)
      << "Input layer should have 1 or 3 channels.";

  input_layer->Reshape(batch_, channels_, height_, width_);
  net_->Reshape();
}

void Predictor::Predict() {
  setMode();

  results_ = nullptr;

#if 1
  StartProfile<float> *start_profile = nullptr;
  EndProfile<float> *end_profile = nullptr;
  if (prof_ != nullptr && profile_enabled_ == false) {
    start_profile = new StartProfile<float>(prof_, net_);
    end_profile = new EndProfile<float>(prof_);
    net_->add_before_forward(start_profile);
    net_->add_after_forward(end_profile);
    profile_enabled_ = true;
  }
#endif

  // net_->set_debug_info(true);
  const auto rr = net_->Forward(bottom);

  for (int ii = 0; ii < output_indexes_.size(); ii++) {
    const auto output_layer = net_->output_blobs()[output_indexes_[ii]];
    pred_len_ += output_layer->height();
    results_[ii] = output_layer->cpu_data();
  }
}

void Predictor::SetInput(int index, void *data, size_t size) {
  auto blob = new caffe::Blob<float>(batch_, channels_, height_, width_);

  if (mode_ == Caffe::CPU) {
    blob->set_cpu_data(inputData);
  } else {
#ifndef CPU_ONLY
    blob->set_gpu_data(inputData);
    blob->mutable_gpu_data();
#endif
  }

  const std::vector<caffe::Blob<float> *> bottom{blob};
}

size_t Predictor::GetOutputSize(int index) {}

void Predictor::GetOutputData(int index) {}

PredictorContext NewCaffe(char *model_file, char *trained_file, int batch_size,
                          std::vector<int> output_indexes, int mode) {
  try {
    const auto ctx = new Predictor(model_file, trained_file, batch_size,
                                   output_indexes, (caffe::Caffe::Brew)mode);
    return (void *)ctx;
  } catch (const std::invalid_argument &ex) {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return nullptr;
  }
}

void SetModeCaffe(int mode) {
  static bool mode_set = false;
  if (!mode_set) {
    mode_set = true;
    Caffe::set_mode((caffe::Caffe::Brew)mode);
    if (mode == Caffe::Brew::GPU) {
      Caffe::SetDevice(0);
    }
  }
}

void InitCaffe() { ::google::InitGoogleLogging("go-caffe"); }

void PredictCaffe(PredictorContext pred, float *inputData) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  predictor->Predict(inputData);
  return;
}

const std::vector<float *> GetPredictionsCaffe(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return nullptr;
  }
  return predictor->results_;
}

void DeleteCaffe(PredictorContext pred) {
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

void StartProfilingCaffe(PredictorContext pred, const char *name,
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

void EndProfilingCaffe(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->end();
  }
}

void DisableProfilingCaffe(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return;
  }
  if (predictor->prof_) {
    predictor->prof_->reset();
  }
}

char *ReadProfileCaffe(PredictorContext pred) {
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

int GetWidthCaffe(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->width_;
}

int GetHeightCaffe(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->height_;
}

int GetChannelsCaffe(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->channels_;
}

int GetPredLenCaffe(PredictorContext pred) {
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    return 0;
  }
  return predictor->pred_len_;
}
