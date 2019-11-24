import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm  #_notebook as tqdm
import matplotlib.pyplot as plt
import gc
import torchvision.transforms as transforms
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
# from src.DataProcessing import DataProcessing
from DataLoading import DataLoading

class Routine(object):
    def __init__(self, settings, model):
        self._settings = settings
        self.train_data = None
        self.test_data = None
        self.path = self._settings.path
        self.device = self._settings.device
        self.Model = model
        self.dataset = DataLoading
        self.xzy_slope = None
        # self.data_ulti = DataProcessing

    def run(self):
        train_images_dir = self.path + 'train_images/{}.jpg'
        test_images_dir = self.path + 'test_images/{}.jpg'
        self.load_data(self._settings.data_sample_factor)
        df_train, df_dev = train_test_split(self.train_data, test_size=0.1, random_state=42)
        df_test = self.test_data

        # Create dataset objects
        transform = transforms.Compose([transforms.ToTensor, transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        train_dataset = DataLoading(self._settings, df_train, train_images_dir, training=True)
        dev_dataset = DataLoading(self._settings, df_dev, train_images_dir, training=False)
        test_dataset = DataLoading(self._settings, df_test, test_images_dir, training=False)

        train_loader = DataLoader(dataset=train_dataset, batch_size=self._settings.batch_size, shuffle=True, num_workers=0)
        dev_loader = DataLoader(dataset=dev_dataset, batch_size=self._settings.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self._settings.batch_size, shuffle=False, num_workers=0)

        model = self.Model(8, self._settings).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self._settings.lr)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(self._settings.epoch, 10) * len(train_loader) // 3, gamma=0.1)

        history = pd.DataFrame()
        for epoch in range(self._settings.epoch):
            self.train_model(optimizer, model, exp_lr_scheduler, epoch, train_loader, history)
            self.evaluate_model(model, epoch, dev_loader, history)
        torch.save(model.state_dict(), './model.pth')
        history['train_loss'].iloc[100:].plot()
        series = history.dropna()['dev_loss']
        plt.scatter(series.index, series)

        torch.cuda.empty_cache()
        gc.collect()

        dev_dataset.train_dataset = self.train_data
        dev_dataset.point_regr()
        #for idx in range(8):
        #    img, mask, regr = dev_dataset[idx]
        #    dev_dataset.show_result(idx, model, img, mask, regr, df_dev)
        predictions = []
        model.eval()
        for img, _, _ in tqdm(test_loader):
            with torch.no_grad():
                output = model(img.to(self.device))
            output = output.data.cpu().numpy()
            for out in output:
                coords = dev_dataset.extract_coords(out)
                s = dev_dataset.coords2str(coords)
                predictions.append(s)
        test = pd.read_csv(self.path + 'sample_submission.csv')
        test['PredictionString'] = predictions
        test.to_csv('predictions.csv', index=False)
        test.head()

    def criterion(self, prediction, mask, regr, size_average=True):
        # Binary mask loss
        pred_mask = torch.sigmoid(prediction[:, 0])
        #     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
        mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
        mask_loss = -mask_loss.mean(0).sum()

        # Regression L1 loss
        pred_regr = prediction[:, 1:]
        regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
        regr_loss = regr_loss.mean(0)

        # Sum
        loss = mask_loss + regr_loss
        if not size_average:
            loss *= prediction.shape[0]
        return loss

    def train_model(self, optimizer, model, exp_lr_scheduler, epoch, train_loader, history=None):
        model.train()
        for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
            img_batch = img_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)
            regr_batch = regr_batch.to(self.device)

            optimizer.zero_grad()
            try:
                output = model(img_batch)
            except RuntimeError as exception:
                if 'out of memory' in str(exception):
                    print('WARNING: out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            loss = self.criterion(output, mask_batch, regr_batch)
            if history is not None:
                history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()

            loss.backward()

            optimizer.step()
            exp_lr_scheduler.step()

        print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
            epoch,
            optimizer.state_dict()['param_groups'][0]['lr'],
            loss.data))

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

    def load_data(self, sample_factor=1):
        train = pd.read_csv(self.path + 'train.csv')
        print(train.shape)
        test = pd.read_csv(self.path + 'sample_submission.csv')

        # From camera.zip
        camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                                       [0, 2305.8757, 1354.9849],
                                       [0, 0, 1]], dtype=np.float32)
        camera_matrix_inv = np.linalg.inv(camera_matrix)
        self.train_data = train.sample(frac=sample_factor, random_state=1, axis=0)
        print(self.train_data.shape)
        self.test_data = test