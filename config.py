# Modifications for 2nd version:
# 1. EMB_DIM = 100 from 50
# 2. Epoch = 15 from 10
# 3. MAX_LEN = 150 from 180
# 4. sample weights for validation
# 5. Multiple kernel size for char CNN
# 6. char filters number to 32 from 64

# 3rd modifications:
# 1. character CNN set all (zero)s for padding 0.

# Questions: 
# 1. forms
# 2. data imbalance
# 3. subword and character CNN to address out of vocabulary
# 4. hyperparameters choice
# 5. MAX_LEN
# 6. use mask to ignore padding
# 7. final metrics
# 8. Word2Vec choice
# 9. focal loss to improve model
# 10. appendix in project(the end is for what, code?)
# 11. Increase window size of Word2Vec 


EMB_DIM = 50 # Word Embedding Dimension
WINDOW_SIZE = 3
KEY_DIM = 64 # Attention Matrix Dimension
NUM_OF_CONV_LAYERS = 128 # Number of Layers Used in Convolution Layer
DROPOUT_RATE = 0.2
FULL_MODEL_PATH = "outputs/Private_Info_Detector_Model.keras"
WV_MODEL_PATH = "outputs/my_w2v_model.model"
WV_META_PATH = "outputs/wv_model_meta.json"
EPOCH_SIZE = 15
MAX_CHAR_LEN = 20
CHAR_DIM = 32 # Character embedding dimension
CHAR_VOCAB_PATH = "outputs/char_vocab.json"
META_PATH = "outputs/model_meta.json"
BATCH_SIZE = 64
CHAR_CNN_FILTERS = 32
LABEL_MAP_PATH = "outputs/label2id.json"
NUM_CLASSES_PATH = "outputs/num_classes.json"
MAX_LEN = 100
PRED_CACHE_PATH = "outputs/pred_cache_test.npz"

COMPARISON_CSV_PATH = "outputs/model_comparison.csv"
COMPARISON_PNG_PATH = "outputs/model_comparison.png"
WITHOUT_CHAR_MODEL_PATH = "outputs/model_without_char_cnn.keras"
WITHOUT_ATTENTION_MODEL_PATH = "outputs/model_without_attention.keras"
PREPARED_DATA_PATH = "outputs/prepared_data_bundle.pkl"