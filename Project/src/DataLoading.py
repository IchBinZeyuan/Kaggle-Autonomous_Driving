import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
from math import sin, cos

import torch


class DataLoading(object):
    def __init__(self, settings, dataframe, root_dir, transform=None):
        self._settings = settings
        self.path = self._settings.path
        self.camera_matrix = self.camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                                       [0, 2305.8757, 1354.9849],
                                       [0, 0, 1]], dtype=np.float32)
        self.IMG_WIDTH = self._settings.img_width
        self.IMG_HEIGHT = self.IMG_WIDTH // 16 * 5
        self.MODEL_SCALE = self._settings.model_scale
        self.DISTANCE_THRESH_CLEAR = self._settings.distance_thresh_clear
        self.IMG_SHAPE = None

        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = self._settings.mode
        self.train_dataset = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Augmentation
        flip = False
        if self.training == 'TRAINING':
            flip = np.random.randint(10) == 1

        # Read image
        img0 = self.imread(img_name, True)
        img = self.preprocess_image(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)

        # Get mask and regression maps
        mask, regr = self.get_mask_and_regr(img0, labels, flip=False)
        regr = np.rollaxis(regr, 2, 0)

        return [img, mask, regr]

    def load_data(self, sample_factor=1):
        self.train_dataset = pd.read_csv(self.path + 'train.csv')
        print(self.train_dataset.shape)
        test = pd.read_csv(self.path + 'sample_submission.csv')

        # From camera.zip
        self.camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                                  [0, 2305.8757, 1354.9849],
                                  [0, 0, 1]], dtype=np.float32)
        camera_matrix_inv = np.linalg.inv(self.camera_matrix)
        self.train_dataset = self.train_dataset.sample(frac=sample_factor, random_state=1, axis=0)
        print(self.train_dataset.shape)
        # return self.train_dataset

    def imread(self, path, fast_mode=False):
        img = cv2.imread(path)
        if not fast_mode and img is not None and len(img.shape) == 3:
            img = np.array(img[:, :, ::-1])  # inverse load image
        return img

    def preprocess_image(self, img, flip=False):
        img = img[img.shape[0] // 2:]
        bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
        bg = bg[:, :img.shape[1] // 6]
        img = np.concatenate([bg, img, bg], 1)
        img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
        if flip:
            img = img[:, ::-1]
        return (img / 255).astype('float32')

    def get_mask_and_regr(self, img, labels, flip=False):
        mask = np.zeros([self.IMG_HEIGHT // self.MODEL_SCALE, self.IMG_WIDTH // self.MODEL_SCALE], dtype='float32')
        regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
        regr = np.zeros([self.IMG_HEIGHT // self.MODEL_SCALE, self.IMG_WIDTH // self.MODEL_SCALE, 7], dtype='float32')
        coords = self.str2coords(labels)
        xs, ys = self.get_img_coords(labels)
        for x, y, regr_dict in zip(xs, ys, coords):
            x, y = y, x
            x = (x - img.shape[0] // 2) * self.IMG_HEIGHT / (img.shape[0] // 2) / self.MODEL_SCALE
            x = np.round(x).astype('int')
            y = (y + img.shape[1] // 6) * self.IMG_WIDTH / (img.shape[1] * 4 / 3) / self.MODEL_SCALE
            y = np.round(y).astype('int')
            if x >= 0 and x < self.IMG_HEIGHT // self.MODEL_SCALE and y >= 0 and y < self.IMG_WIDTH // self.MODEL_SCALE:
                mask[x, y] = 1
                regr_dict = self._regr_preprocess(regr_dict, flip)
                regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
        if flip:
            mask = np.array(mask[:, ::-1])
            regr = np.array(regr[:, ::-1])
        return mask, regr

    def str2coords(self, s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
        '''
        Input:
            s: PredictionString (e.g. from train dataframe)
            names: array of what to extract from the string
        Output:
            list of dicts with keys from `names`
        '''
        coords = []
        for l in np.array(s.split()).reshape([-1, 7]):
            coords.append(dict(zip(names, l.astype('float'))))
            if 'id' in coords[-1]:
                coords[-1]['id'] = int(coords[-1]['id'])  # change id to int dtype
        return coords

    def get_img_coords(self, s):
        '''
        Input is a PredictionString (e.g. from train dataframe)
        Output is two arrays:
            xs: x coordinates in the image
            ys: y coordinates in the image
        '''
        coords = self.str2coords(s)
        xs = [c['x'] for c in coords]
        ys = [c['y'] for c in coords]
        zs = [c['z'] for c in coords]
        P = np.array(list(zip(xs, ys, zs))).T  # row: x, y, z
        img_p = np.dot(self.camera_matrix, P).T
        img_p[:, 0] /= img_p[:, 2]
        img_p[:, 1] /= img_p[:, 2]
        img_xs = img_p[:, 0]
        img_ys = img_p[:, 1]
        img_zs = img_p[:, 2]  # z = Distance from the camera
        return img_xs, img_ys

    def _regr_preprocess(self, regr_dict, flip=False):
        if flip:
            for k in ['x', 'pitch', 'roll']:
                regr_dict[k] = -regr_dict[k]
        for name in ['x', 'y', 'z']:
            regr_dict[name] = regr_dict[name] / 100
        regr_dict['roll'] = self.rotate(regr_dict['roll'], np.pi)
        regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
        regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
        regr_dict.pop('pitch')
        regr_dict.pop('id')
        return regr_dict

    def rotate(self, x, angle):  # rotate angle
        x = x + angle
        x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
        return x

    # convert euler angle to rotation matrix
    def euler_to_Rot(self, yaw, pitch, roll):
        Y = np.array([[cos(yaw), 0, sin(yaw)],
                      [0, 1, 0],
                      [-sin(yaw), 0, cos(yaw)]])
        P = np.array([[1, 0, 0],
                      [0, cos(pitch), -sin(pitch)],
                      [0, sin(pitch), cos(pitch)]])
        R = np.array([[cos(roll), -sin(roll), 0],
                      [sin(roll), cos(roll), 0],
                      [0, 0, 1]])
        return np.dot(Y, np.dot(P, R))

    def draw_line(self, image, points):
        color = (255, 0, 0)
        cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
        cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
        cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
        cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
        return image

    def draw_points(self, image, points):
        for (p_x, p_y, p_z) in points:
            cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
        #         if p_x > image.shape[1] or p_y > image.shape[0]:
        #             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
        return image

    def visualize(self, img, coords):
        # You will also need functions from the previous cells
        x_l = 1.02
        y_l = 0.80
        z_l = 2.31

        img = img.copy()
        for point in coords:
            # Get values
            x, y, z = point['x'], point['y'], point['z']
            yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
            # Math
            Rt = np.eye(4)  # 4*4对角矩阵
            t = np.array([x, y, z])
            Rt[:3, 3] = t
            Rt[:3, :3] = self.euler_to_Rot(yaw, pitch, roll).T
            Rt = Rt[:3, :]
            P = np.array([[x_l, -y_l, -z_l, 1],
                          [x_l, -y_l, z_l, 1],
                          [-x_l, -y_l, z_l, 1],
                          [-x_l, -y_l, -z_l, 1],
                          [0, 0, 0, 1]]).T
            img_cor_points = np.dot(self.camera_matrix, np.dot(Rt, P))
            img_cor_points = img_cor_points.T
            img_cor_points[:, 0] /= img_cor_points[:, 2]
            img_cor_points[:, 1] /= img_cor_points[:, 2]
            img_cor_points = img_cor_points.astype(int)
            # Drawing
            img = self.draw_line(img, img_cor_points)
            img = self.draw_points(img, img_cor_points[-1:])

        return img

    def _regr_back(self, regr_dict):
        for name in ['x', 'y', 'z']:
            regr_dict[name] = regr_dict[name] * 100
        regr_dict['roll'] = self.rotate(regr_dict['roll'], -np.pi)

        pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
        pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
        regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
        return regr_dict


    def convert_3d_to_2d(self, x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
        # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
        return x * fx / z + cx, y * fy / z + cy

    def optimize_xy(self, r, c, x0, y0, z0, flipped=False):
        def distance_fn(xyz):
            img = self.imread(self.path + 'train_images/ID_8a6e65317' + '.jpg')
            IMG_SHAPE = img.shape
            x, y, z = xyz
            xx = -x if flipped else x
            slope_err = (self.point_regr().predict([[xx, z]])[0] - y) ** 2
            x, y = self.convert_3d_to_2d(x, y, z0)
            y, x = x, y
            x = (x - IMG_SHAPE[0] // 2) * self.IMG_HEIGHT / (IMG_SHAPE[0] // 2) / self.MODEL_SCALE
            # x = np.round(x).astype('int')
            y = (y + IMG_SHAPE[1] // 6) * self.IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / self.MODEL_SCALE
            # y = np.round(y).astype('int')
            # return (x - r) ** 2 + (y - c) ** 2
            return max(0.2, (x - r) ** 2 + (y - c) ** 2) + max(0.4, slope_err)

        res = minimize(distance_fn, [x0, y0, z0], method='Powell')
        x_new, y_new, z_new = res.x
        return x_new, y_new, z_new

    def clear_duplicates(self, coords):
        for c1 in coords:
            xyz1 = np.array([c1['x'], c1['y'], c1['z']])
            for c2 in coords:
                xyz2 = np.array([c2['x'], c2['y'], c2['z']])
                distance = np.sqrt(((xyz1 - xyz2) ** 2).sum())
                if distance < self.DISTANCE_THRESH_CLEAR:
                    if c1['confidence'] < c2['confidence']:
                        c1['confidence'] = -1
        return [c for c in coords if c['confidence'] > 0]

    def extract_coords(self, prediction, flipped=False):
        logits = prediction[0]
        regr_output = prediction[1:]
        points = np.argwhere(logits > 0)
        print('check points', points.shape)
        col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
        coords = []
        for r, c in points:
            regr_dict = dict(zip(col_names, regr_output[:, r, c]))
            coords.append(self._regr_back(regr_dict))
            coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
            coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = self.optimize_xy(r, c, coords[-1]['x'], coords[-1]['y'],
                                                                            coords[-1]['z'], flipped)
        coords = self.clear_duplicates(coords)
        print('check coords', coords)
        return coords

    def coords2str(self, coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
        s = []
        for c in coords:
            for n in names:
                s.append(str(c.get(n, 0)))
        return ' '.join(s)

    def point_regr(self):
        #self.load_data()

        points_df = pd.DataFrame()
        for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
            arr = []
            for ps in self.train_dataset['PredictionString']:
                coords = self.str2coords(ps)
                arr += [c[col] for c in coords]
            points_df[col] = arr

        zy_slope = LinearRegression()
        X = points_df[['z']]
        y = points_df['y']
        zy_slope.fit(X, y)
        #  print('MAE without x:', mean_absolute_error(y, zy_slope.predict(X)))

        # Will use this model later
        xzy_slope = LinearRegression()
        X = points_df[['x', 'z']]
        y = points_df['y']
        xzy_slope.fit(X, y)
        #  print('MAE with x:', mean_absolute_error(y, xzy_slope.predict(X)))

        # print('\ndy/dx = {:.3f}\ndy/dz = {:.3f}'.format(*xzy_slope.coef_))
        return xzy_slope