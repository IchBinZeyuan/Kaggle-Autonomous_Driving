path = '/home/ge34caj/Downloads/pku-autonomous-driving/'
log_path = '/home/ge34caj/Downloads/pku-autonomous-driving/Results/'

mode = 'TRAINING'
device = 'cuda'

epoch = 10

img_width = 512
model_scale = 8
distance_thresh_clear = 2
data_agument = ['flip', 'blur', 'noise']
data_agument_ratio = 0.1

data_sample_factor = 1
validation_data_ratio = 0.05
# nrof_test_data = 100

optimize_coordinate = False

pre_trained = True
lr = 5e-3
lr_decay_epoch = 5
reg_factor = 1e-3
batch_size = 8

save_model = True