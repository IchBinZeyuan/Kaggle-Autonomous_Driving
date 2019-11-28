path = '/home/ge34caj/Downloads/pku-autonomous-driving/'
log_path = '/home/ge34caj/Downloads/pku-autonomous-driving/Results/'

mode = 'TRAINING'
device = 'cuda'

epoch = 10

img_width = 1024
model_scale = 8
distance_thresh_clear = 2

data_sample_factor = 1
test_data_ratio = 0.1

pre_trained = False
lr = 1e-3
reg_factor = 1e-4
batch_size = 2

save_model = True