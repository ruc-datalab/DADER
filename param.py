# params for dataset and data loader
data_root = "data"

# params for source dataset
src_encoder_path = "source-encoder.pt"
src_classifier_path = "source-classifier.pt"

# params for target dataset
tgt_encoder_path = "target-encoder.pt"
tgt_classifier_path = "target-classifier.pt"
model_root = "checkpoint"
d_model_path = "critic.pt"
mlp_model_path = "mlp.pt"

# params for training network
c_learning_rate = 2e-5
d_learning_rate = 1e-5
weight_decay = 1e-6
hidden_size = 768
intermediate_size = 3072
kernel_num = 20
kernel_sizes = [3, 4, 5]
pretrain = True
dropout = 0.1
num_labels = 2
h_dim = 300

"""Special tokens."""
PAD = 0
UNK = 1
SOS = 2
EOS = 3
