#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data, Dtype* temp_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const bool rotate = param_.rotate();
  const Dtype scale = param_.scale();
  const int window_size = param_.window_size();
  
  int newHeight = datum.height();
  int newWidth = datum.width();
  
  if(crop_size) {
	newHeight = crop_size;
	newWidth = crop_size;
  } else if(window_size) {
    newHeight = window_size;
	newWidth = window_size;
  }
  
  
  if (window_size && crop_size) {
    LOG(FATAL) << "Current implementation does not support window_size and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size || window_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN && crop_size) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - newHeight) / 2;
      w_off = (width - newWidth) / 2;
    }
    
	// Normal copy
	for (int c = 0; c < channels; ++c) {
		for (int h = 0; h < newHeight; ++h) {
		  for (int w = 0; w < newWidth; ++w) {
			int top_index = ((batch_item_id * channels + c) * newHeight + h)
				* newWidth + w;
			int data_index = (c * height + h + h_off) * width + w + w_off;
			Dtype datum_element =
				static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
			transformed_data[top_index] =
				(datum_element - mean[data_index]) * scale;
		  }
		}
	}  
  } else {
	//LOG(INFO) << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Normal::"  << batch_item_id;
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
  
  //Perform mirroring on the transformed_data using a temp_data first then copy it back
  if (mirror && Rand() % 3) {
      // Copy mirrored version
	  if(Rand()%2){ //Mirror vertical
		//LOG(INFO) << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Mirror vertical::" << batch_item_id;
        for (int c = 0; c < channels; ++c) {
         for (int h = 0; h < newHeight; ++h) {
          for (int w = 0; w < newWidth; ++w) {
            int data_index = ((batch_item_id * channels + c) * newHeight + h) * newWidth + w;
			int	top_index = ((batch_item_id * channels + c) * newHeight + h)
					* newWidth + (newWidth - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(transformed_data[data_index]));
            temp_data[top_index] = datum_element;
          }
         }
        } 
	   }else{ //Mirror horizontal
			//LOG(INFO) << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Mirror horizontal::" << batch_item_id;
			for (int c = 0; c < channels; ++c) {
			 for (int h = 0; h < newHeight; ++h) {
			  for (int w = 0; w < newWidth; ++w) {
				int data_index = ((batch_item_id * channels + c) * newHeight + h) * newWidth + w;
				int	top_index = ((batch_item_id * channels + c) * newHeight + (newHeight - 1 -h))
					* newWidth + w;
				Dtype datum_element =
					static_cast<Dtype>(static_cast<uint8_t>(transformed_data[data_index]));
				temp_data[top_index] = datum_element;
			  }
			 }
			} 
		}
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < newHeight; ++h) {
				for (int w = 0; w < newWidth; ++w) {
					int top_index = ((batch_item_id * channels + c) * newHeight + h)
						* newWidth + w;
					Dtype datum_element =
						static_cast<Dtype>(static_cast<uint8_t>(temp_data[top_index]));
					transformed_data[top_index] = datum_element;
				}
			}
		}
    }
  
  
  
	// Perform rotation on the transformed_data using a temp_data first then copy it back
	if(rotate && Rand() %3) {
		int r = Rand() % 2;
		if(r == 0) {//Rotate 90
			//LOG(INFO) << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Rotate 90::" << batch_item_id;
			for (int c = 0; c < channels; ++c) {
				for (int h = 0; h < newHeight; ++h) {
					for (int w = 0; w < newWidth; ++w) {
						int top_index = ((batch_item_id * channels + c) * newHeight + h)
							* newWidth + w;
						int new_top_index = ((batch_item_id * channels + c) * newHeight * newWidth) + h + (newWidth - 1 -w) * newWidth;
						Dtype datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(transformed_data[top_index]));
						temp_data[new_top_index] = datum_element;
					}
				}
			}	
		}else if(r ==1) { //Rotate -90
			//LOG(INFO) << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Rotate -90::" << batch_item_id;
			for (int c = 0; c < channels; ++c) {
				for (int h = 0; h < newHeight; ++h) {
					for (int w = 0; w < newWidth; ++w) {
						int top_index = ((batch_item_id * channels + c) * newHeight + h)
							* newWidth + w;
						int new_top_index = ((batch_item_id * channels + c) * newHeight * newWidth) + (newWidth - 1 -h) + (w * newWidth);
						Dtype datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(transformed_data[top_index]));
						temp_data[new_top_index] = datum_element;
					}
				}
			}
		} else if(r ==2) { //Rotate 180
			for (int c = 0; c < channels; ++c) {
				for (int h = 0; h < newHeight; ++h) {
					for (int w = 0; w < newWidth; ++w) {
						int top_index = ((batch_item_id * channels + c) * newHeight + h)
							* newWidth + w;
						int new_top_index = ((batch_item_id * channels + c) * newHeight + (newHeight-h-1)) * newWidth + (newWidth -w-1);
						Dtype datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(transformed_data[top_index]));
						temp_data[new_top_index] = datum_element;
					}
				}
			}
		}
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < newHeight; ++h) {
				for (int w = 0; w < newWidth; ++w) {
					int top_index = ((batch_item_id * channels + c) * newHeight + h)
						* newWidth + w;
					Dtype datum_element =
						static_cast<Dtype>(static_cast<uint8_t>(temp_data[top_index]));
					transformed_data[top_index] = datum_element;
				}
			}
		}
	}
	//LOG(INFO) << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> -------------------::" << batch_item_id;
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
