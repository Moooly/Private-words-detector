# # 1. Hyperparameter choice(word embedding and character embedding dimension)
# # 4. imbalance 10000 why unstable
# # 5. attention mask (B, 1, 1, T)
## 6. how does weight works

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
from utils.runtime_utils import configure_runtime
from pipelines.training_runner import train_and_save_model
from pipelines.training_pipeline import prepare_data

def main():
    configure_runtime()
    train_size = 0.7
    test_size = 0.3
    data_bundle = prepare_data(train_size, test_size)
    train_and_save_model(data_bundle)


if __name__ == "__main__":
    main()