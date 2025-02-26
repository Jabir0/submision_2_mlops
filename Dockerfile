# Gunakan TensorFlow Serving sebagai base image
FROM tensorflow/serving:latest

# Salin model ke dalam container
COPY ./serving_model /models/hearts_model

# Tentukan environment variable
ENV MODEL_NAME=hearts_model
ENV MODEL_BASE_PATH=/models
ENV MODEL_VERSION=1740451872

# Jalankan TensorFlow Serving dengan versi model yang benar
CMD tensorflow_model_server \
    --rest_api_port=8501 \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
    --model_version_policy="specific:${MODEL_VERSION}"
