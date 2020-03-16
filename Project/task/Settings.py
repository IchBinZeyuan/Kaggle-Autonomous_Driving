path = '/home/ge34caj/Downloads/pku-autonomous-driving/'
log_path = '/home/ge34caj/Downloads/pku-autonomous-driving/Results/'

mode = 'TRAINING'
device = 'cuda'

epoch = 15

img_width = 800
model_scale = 8
distance_thresh_clear = 2
data_agument = ['flip', 'blur', 'noise']
# data_agument = []
data_agument_ratio = 0.1

data_sample_factor = 1
validation_data_ratio = 0.05
# nrof_test_data = 100

optimize_coordinate = True

pre_trained = False
lr = 5e-2
lr_decay_epoch = 5
reg_factor = 1e-3
batch_size = 4

focal_loss = False

save_model = False