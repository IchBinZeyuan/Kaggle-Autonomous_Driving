import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm  #_notebook as tqdm
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from src.DataProcessing import DataProcessing
from termcolor import colored
from math import sqrt, acos, pi, sin, cos
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import average_precision_score
import os
from datetime import datetime


class Routine(object):
    def __init__(self, settings, model):
        self._settings = settings
        self.data_loader = DataProcessing
        self.train_data = None
        self.path = self._settings.path
        self.device = self._settings.device
        self.Model = model
        self.xzy_slope = None

    def run(self):
        dir_name = self.make_log_dir()
        df_train, df_dev, df_test, train_loader, dev_loader, test_loader, train_dataset, dev_dataset, test_dataset = self.data_loading()
        self.show_image(train_dataset[0][0])
        model = self.Model(8, self._settings).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self._settings.lr, weight_decay=self._settings.reg_factor)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=self._settings.lr_decay_epoch * len(train_loader), gamma=0.1)

        history = pd.DataFrame()
        for epoch in range(self._settings.epoch):
            self.train_model(optimizer, model, exp_lr_scheduler, epoch, train_loader, history)
            self.evaluate_model(model, epoch, dev_loader, history)
        if self._settings.save_model:
            torch.save(model.state_dict(), dir_name + '/model.pth')
        history['train_loss'].iloc[100:].plot().get_figure().savefig(dir_name + '/train_loss.png')
        plt.cla()
        series = history.dropna()['dev_loss']
        plt.scatter(series.index, series)
        plt.savefig(dir_name + '/dev_loss.png')
        torch.cuda.empty_cache()
        gc.collect()

        dev_dataset.train_dataset = self.train_data
        dev_dataset.point_regr()
        for idx in range(5):
           img, mask, regr = dev_dataset[idx]
           output = model(torch.tensor(img[None]).to(self._settings.device)).data.cpu().numpy()
           coords_pred = dev_dataset.extract_coords(prediction=output[0])
           coords_true = dev_dataset.extract_coords(prediction=np.concatenate([mask[None], regr], 0))
           dev_dataset.show_result(idx, coords_true, coords_pred, df_dev, dir_name)

        predictions = []
        model.eval()
        for img, _, _ in tqdm(test_loader):
            with torch.no_grad():
                output = model(img.to(self.device))
            output = output.data.cpu().numpy()
            for out in output:
                coords = dev_dataset.extract_coords(out)
                s = self.data_loader.coords2str(coords)
                predictions.append(s)

        test = pd.read_csv(self.path + 'sample_submission.csv', nrows=getattr(self._settings, 'nrof_test_data', None))
        test['PredictionString'] = predictions
        test.to_csv(dir_name + '/predictions.csv', index=False)
        test.head()
        # todo: fix bug in local metric calculation function
        # self.local_metric(test)
        plt.show()

    def data_loading(self):
        train = pd.read_csv(self.path + 'train.csv')
        print(colored('Original Train Data Shape:', 'green'), train.shape)
        test = pd.read_csv(self.path + 'sample_submission.csv', nrows=getattr(self._settings, 'nrof_test_data', None))
        print(colored('Original Test Train Data Shape:', 'green'), test.shape)
        camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                                       [0, 2305.8757, 1354.9849],
                                       [0, 0, 1]], dtype=np.float32)
        camera_matrix_inv = np.linalg.inv(camera_matrix)
        train_data = train.sample(frac=self._settings.data_sample_factor, random_state=1, axis=0)
        df_test = test
        self.train_data = train_data
        train_images_dir = self.path + 'train_images/{}.jpg'
        test_images_dir = self.path + 'test_images/{}.jpg'
        # Create dataset objects
        df_train, df_dev = train_test_split(train_data, test_size=self._settings.validation_data_ratio, random_state=6)
        print(colored('Train Data Shape:', 'green'), df_train.shape)
        print(colored('Evaluation Data Shape:', 'green'), df_dev.shape)
        print(colored('Test Data Shape:', 'green'), df_test.shape)
        train_dataset = self.data_loader(self._settings, df_train, train_images_dir, data_agument=self._settings.data_agument, training=True)
        dev_dataset = self.data_loader(self._settings, df_dev, train_images_dir, training=False)
        test_dataset = self.data_loader(self._settings, df_test, test_images_dir, training=False)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self._settings.batch_size, shuffle=True,
                                  num_workers=0)
        dev_loader = DataLoader(dataset=dev_dataset, batch_size=self._settings.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=0)
        return df_train, df_dev, df_test, train_loader, dev_loader, test_loader, train_dataset, dev_dataset, test_dataset

    def criterion(self, prediction, mask, regr, weight=0.4, size_average=True):
        # Binary mask loss
        pred_mask = torch.sigmoid(prediction[:, 0])
        mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
        mask_loss = -mask_loss.mean(0).sum()
        # Regression L1 loss
        # todo: try l2 loss
        pred_regr = prediction[:, 1:]
        regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
        regr_loss = regr_loss.mean(0)
        # Sum
        loss = weight * mask_loss + (1 - weight) * regr_loss
        if not size_average:
            loss *= prediction.shape[0]
        return loss

    def train_model(self, optimizer, model, exp_lr_scheduler, epoch, train_loader, history=None):
        model.train()
        total_loss = 0
        for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
            img_batch = img_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)
            regr_batch = regr_batch.to(self.device)
            optimizer.zero_grad()
            output = model(img_batch)
            loss = self.criterion(output, mask_batch, regr_batch)
            total_loss += loss.data
            if history is not None:
                history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()
            exp_lr_scheduler.step()
        print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr'], total_loss/len(train_loader.dataset)))

    def evaluate_model(self, model, epoch, dev_loader, history=None):
        model.eval()
        loss = 0

        with torch.no_grad():
            for img_batch, mask_batch, regr_batch in dev_loader:
                img_batch = img_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                regr_batch = regr_batch.to(self.device)

                output = model(img_batch)

                loss += self.criterion(output, mask_batch, regr_batch, size_average=False).data

        loss /= len(dev_loader.dataset)

        if history is not None:
            history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()

        print('Dev loss: {:.4f}'.format(loss))

    def local_metric(self, test_prediction):

        thres_tr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
        thres_ro_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

        def TranslationDistance(p, g, abs_dist=False):
            dx = p['x'] - g['x']
            dy = p['y'] - g['y']
            dz = p['z'] - g['z']
            diff0 = (g['x'] ** 2 + g['y'] ** 2 + g['z'] ** 2) ** 0.5
            diff1 = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            if abs_dist:
                diff = diff1
            else:
                diff = diff1 / diff0
            return diff

        def RotationDistance(p, g):
            true = [g['pitch'], g['yaw'], g['roll']]
            pred = [p['pitch'], p['yaw'], p['roll']]
            q1 = R.from_euler('xyz', true)
            q2 = R.from_euler('xyz', pred)
            diff = R.inv(q2) * q1
            W = np.clip(diff.as_quat()[-1], -1., 1.)

            # in the official metrics code:
            # https://www.kaggle.com/c/pku-autonomous-driving/overview/evaluation
            #   return Object3D.RadianToDegree( Math.Acos(diff.W) )
            # this code treat θ and θ+2π differntly.
            # So this should be fixed as follows.
            W = (acos(W) * 360) / pi
            if W > 180:
                W = 360 - W
            return W

        def expand_df(df, PredictionStringCols):
            df = df.dropna().copy()
            df['NumCars'] = [int((x.count(' ') + 1) / 7) for x in df['PredictionString']]

            image_id_expanded = [item for item, count in zip(df['ImageId'], df['NumCars']) for i in range(count)]
            prediction_strings_expanded = df['PredictionString'].str.split(' ', expand=True).values.reshape(-1,
                                                                                                            7).astype(
                float)
            prediction_strings_expanded = prediction_strings_expanded[
                ~np.isnan(prediction_strings_expanded).all(axis=1)]
            df = pd.DataFrame(
                {
                    'ImageId': image_id_expanded,
                    PredictionStringCols[0]: prediction_strings_expanded[:, 0],
                    PredictionStringCols[1]: prediction_strings_expanded[:, 1],
                    PredictionStringCols[2]: prediction_strings_expanded[:, 2],
                    PredictionStringCols[3]: prediction_strings_expanded[:, 3],
                    PredictionStringCols[4]: prediction_strings_expanded[:, 4],
                    PredictionStringCols[5]: prediction_strings_expanded[:, 5],
                    PredictionStringCols[6]: prediction_strings_expanded[:, 6]
                })
            return df

        def check_match(idx):
            keep_gt = False
            thre_tr_dist = thres_tr_list[idx]
            thre_ro_dist = thres_ro_list[idx]
            train_dict = {imgID: self.data_loader.str2coords(s, names=['carid_or_score', 'pitch', 'yaw', 'roll', 'x', 'y', 'z']) for
                          imgID, s in zip(train_df['ImageId'], train_df['PredictionString'])}
            valid_dict = {imgID: self.data_loader.str2coords(s, names=['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'carid_or_score']) for
                          imgID, s in zip(valid_df['ImageId'], valid_df['PredictionString'])}
            result_flg = []  # 1 for TP, 0 for FP
            scores = []
            MAX_VAL = 10 ** 10
            for img_id in valid_dict:
                for pcar in sorted(valid_dict[img_id], key=lambda x: -x['carid_or_score']):
                    # find nearest GT
                    min_tr_dist = MAX_VAL
                    min_idx = -1
                    for idx, gcar in enumerate(train_dict[img_id]):
                        tr_dist = TranslationDistance(pcar, gcar)
                        if tr_dist < min_tr_dist:
                            min_tr_dist = tr_dist
                            min_ro_dist = RotationDistance(pcar, gcar)
                            min_idx = idx

                    # set the result
                    if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                        if not keep_gt:
                            train_dict[img_id].pop(min_idx)
                        result_flg.append(1)
                    else:
                        result_flg.append(0)
                    scores.append(pcar['carid_or_score'])

            return result_flg, scores

        # expanded_valid_df = expand_df(test_prediction, ['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'Score'])
        test_prediction = test_prediction.fillna('')

        valid_df = test_prediction

        train_df = pd.read_csv(self.path + 'train.csv')
        train_df = train_df[train_df.ImageId.isin(test_prediction.ImageId.unique())]
        # data description page says, The pose information is formatted as
        # model type, yaw, pitch, roll, x, y, z
        # but it doesn't, and it should be
        # model type, pitch, yaw, roll, x, y, z
        expanded_train_df = expand_df(train_df, ['model_type', 'pitch', 'yaw', 'roll', 'x', 'y', 'z'])

        max_workers = 10
        n_gt = len(expanded_train_df)
        ap_list = []
        # p = Pool(processes=max_workers)
        for result_flg, scores in (check_match(i) for i in range(10)):
            n_tp = np.sum(result_flg)
            recall = n_tp / n_gt
            ap = average_precision_score(result_flg, scores) * recall
            ap_list.append(ap)
        map = np.mean(ap_list)
        print(colored('mAP score is :', 'blue'), map)

    @staticmethod
    def show_image(image):
        # plt.ion()
        plt.figure()
        plt.title('processed image')
        img = image
        img = np.rollaxis(img, 0, 3)
        plt.imshow(img)
        plt.show()
        # plt.pause(0.01)
        # plt.clf()

    def make_log_dir(self):
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        repository_name = self._settings.log_path + dt_string + '_' + str(self._settings.batch_size)
        os.makedirs(repository_name)
        return repository_name

