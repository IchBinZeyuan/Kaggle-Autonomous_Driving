path = '/home/ge34caj/Downloads/pku-autonomous-driving/'
log_path = '/home/ge34caj/Downloads/pku-autonomous-driving/Results/'

mode = 'TRAINING'
device = 'cuda'

epoch = 8

img_width = 1024
model_scale = 8
distance_thresh_clear = 2
data_agument = ['flip', 'blur']
data_agument_ratio = 0.1

data_sample_factor = 1
validation_data_ratio = 0.05
# nrof_test_data = 100

pre_trained = True
lr = 5e-3
lr_decay_epoch = 3
reg_factor = 1e-3
batch_size = 2

save_model = True