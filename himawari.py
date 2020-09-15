import os
import sys
from pyhdf.SD import SD, SDC

from scipy import ndimage
import numpy as np
import pandas as pd
import tensorflow as tf

import glob


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def make_patches(img, size, stride):
    patches = []
    for i in np.arange(0, img.shape[0], stride):
        if (i+size) > img.shape[0]:
            i = img.shape[0] - size
        for j in np.arange(0, img.shape[1], stride):
            if (j+size) > img.shape[1]:
                j = img.shape[1] - size
            patches.append(img[i:i+size,j:j+size])
    return patches


### LOAD SINGLE MAIAC FILE ####
def load_AHI12(AHI12_file):
#    print("AHI12_file", AHI12_file)
    fp = SD(AHI12_file, SDC.READ)
    ahi12_obj = fp.select('sur_refl')

    # Read AHI12 Data - SR
    AHI12_data = ahi12_obj.get()[:6]
    AHI12_data = np.moveaxis(AHI12_data, 0, -1)

    # rescale and replace NaN's with -1
    AHI12_data = AHI12_data.astype(float) * 1e-4
    AHI12_data[AHI12_data < 0] = -1
    

    return AHI12_data


### LOAD SINGLE HIMAWARI FILE ###
def load_AHI05(AHI05_file, bands, solar=False):
    AHI05_data = []
    fp = SD(AHI05_file, SDC.READ)
    for band in bands:
        ahi12_obj = fp.select('AHI_B%02i' % band)
        arr = ahi12_obj.get()[:]
        if arr.shape[0] != 600:
            arr = ndimage.interpolation.zoom(arr, 600./arr.shape[0], order=1)
        AHI05_data.append(np.expand_dims(arr, 2))

    AHI05_data = np.concatenate(AHI05_data, 2)
    AHI05_data = AHI05_data.astype(float) * 1e-4
    AHI05_data[AHI05_data < 0] = -1

    if solar:
        sa = fp.select('Solar_Azimuth').get()[:]
        sz = fp.select('Solar_Zenith').get()[:]
        s = np.concatenate([np.expand_dims(sz, 2), np.expand_dims(sa, 2)], 2)
        s = s * 0.01
        AHI05_data = np.concatenate((AHI05_data, s), axis=2)

    return AHI05_data


### WRITE DATA IN BATCHES TO TFRECORDS ### 
class TFRecordWriter:
    def __init__(self, dir, chunksize=500, finite_ratio=0.75):
        self.dir = dir
        self.chunksize = chunksize
        self.counter = 1
        self.filecounter = 0
        self.skipped = 0
        self.finite_pixels = 0
        self.finite_ratio = finite_ratio

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.filename = os.path.join(self.dir, 'part-%05i.tfrecords')
        print("Tensorflow version", tf.__version__)
        self.writer = tf.python_io.TFRecordWriter(self.filename % self.filecounter)

    def write_records(self, m):
        n, _, _, _ = m['band1_0500'].shape
        idxs = np.arange(n)
        np.random.shuffle(idxs)

        for i in idxs:
            if self.counter % self.chunksize == 0:
                self.writer.close()
                self.filecounter += 1
                self.writer = tf.python_io.TFRecordWriter(self.filename % self.filecounter)

            #features = {'height-250': _int64_feature(h),
            #            'width-250': _int64_feature(w)}
            features = dict()
            skip_example = False
            for k, item in m.items():
                x = item[i]
                finitevals = np.mean(np.isfinite(x))
                if finitevals < self.finite_ratio:
                    skip_example = True
                    break
                x[np.isnan(x)] = -9999.
                features[k] = _bytes_feature(item[i].astype(np.float32).tostring())

            if not skip_example:
                example = tf.train.Example(features=tf.train.Features(feature=features))
                self.writer.write(example.SerializeToString())
                self.counter += 1
            else:
                self.skipped += 1

        self.keys = features.keys()


### READING DATA ON NAS ###
class NASHimawari:
    def __init__(self, data_directory='/nobackupp10/nexprojects/GEONEX/datapool/HM08_AHI_v1'):
        self.data_directory = data_directory

    def AHI12_files(self, tile=None, year=None, dayofyear=None):
        dir = os.path.join(self.data_directory,'HM08_AHI12')
        if tile == None:
            tile = '*'
        if year == None:
            year = '*'
        else:
            year = str(year)
        if dayofyear == None:
            dayofyear = '*'
        else:
            dayofyear = '%03i' % dayofyear

        file_pattern = os.path.join(dir, '%s/%s/HM08_AHI12B_%s%s*.hdf' % (tile, year, year, dayofyear))
        files = glob.glob(file_pattern)

        fileinfo = []
        for f in files:
            root = os.path.dirname(f)
            rl = root.split('/')

            y = int(rl[-1])
            tile = rl[-2]

            fl = os.path.basename(f).split('_')
            doy = int(fl[2][4:7])

            hour = int(fl[2][7:9])
            minute = int(fl[2][9:11])
            fileinfo.append(dict(year=y, dayofyear=doy, hour=hour,
                              minute=minute, ahi12_file=f, tile=tile))
        fileinfo = pd.DataFrame(fileinfo)
        
        if len(fileinfo) == 0:
            print('no AHI12 files found for tile {} year {}'.format(tile, year))
            fileinfo = pd.DataFrame(columns=['year', 'dayofyear', 'hour',
                                             'minute', 'ahi12_file', 'tile'])
        return fileinfo
   

    def AHI05_files(self, tile=None, year=None, dayofyear=None):
        dir = os.path.join(self.data_directory,'HM08_AHI05')
        if tile == None:
            tile = '*'
        if year == None:
            year = '*'
        else:
            year = str(year)
        if dayofyear == None:
            dayofyear = '*'
        else:
            dayofyear = '%03i' % dayofyear

        file_pattern = os.path.join(dir, '%s/%s/%s/*.hdf' % (tile, year, dayofyear))
        files = glob.glob(file_pattern)

        fileinfo = []
        for f in files:
            root = os.path.dirname(f)
            rl = root.split('/')
            doy = int(rl[-1])
            y = int(rl[-2])

            fl = os.path.basename(f).split('_')
            hour = int(fl[3][:2])
            minute = int(fl[3][2:])
            tile = fl[5]
            fileinfo.append(dict(year=y, dayofyear=doy, hour=hour,
                              minute=minute, ahi05_file=f, tile=tile))
        fileinfo = pd.DataFrame(fileinfo)
        
        if len(fileinfo) == 0:
            print('no AHI05 files found for tile {} year {}'.format(tile, year))
            fileinfo = pd.DataFrame(columns=['year', 'dayofyear', 'hour',
                                             'minute', 'ahi05_file', 'tile'])
        return fileinfo

    def file_pairs(self, tile=None, year=None, dayofyear=None):
        print("Getting AHI12 Files")
        ahi12_files = self.AHI12_files(tile=tile, year=year, dayofyear=dayofyear)
        
        print("Getting AHI05 Files")
        ahi05_files = self.AHI05_files(tile=tile, year=year, dayofyear=dayofyear)
        
        file_df = ahi12_files.merge(ahi05_files,
            left_on=['year', 'dayofyear', 'hour', 'minute', 'tile'],
            right_on=['year', 'dayofyear', 'hour', 'minute', 'tile'])

        return file_df


### MAIAC EMULATOR DATA PIPELINE DEPENDING ON NAS ### 
class EmulatorData:
    def __init__(self, nas_dir='/nobackupp10/nexprojects/GEONEX/datapool/HM08_AHI_v1',
                       tf_data_dir='/nobackupp13/kmduffy1/training',
                       input_bands=[1] + list(range(3,17)),
                       patch_size=50,
                       patch_stride=40,
                       coverage_floor=0.50):
        
        self.nas = NASHimawari(nas_dir)
        self.tf_data_dir = tf_data_dir
        if not os.path.exists(self.tf_data_dir):
            os.makedirs(self.tf_data_dir)

        self.input_bands = input_bands
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.coverage_floor = coverage_floor

    def file_pairs(self, tile=None, year=None, dayofyear=None):
        return self.nas.file_pairs(tile, year, dayofyear)

    def load_files(self, AHI05_file, AHI12_file, solar=False):
        # Read each band of AHI05
        AHI05_data = load_AHI05(AHI05_file, self.input_bands, solar=solar)
        AHI12_data = load_AHI12(AHI12_file)
        
        # make mask
        mask = np.ones_like(AHI12_data)
        mask[AHI12_data < 0] = 0
        mask = mask[:,:,0]
        return AHI05_data, AHI12_data, mask

    def check_coverage(self, img):
        total_pixels = float(len(img.flatten()))
        nodata_pixels = float(list(img.flatten()).count(0))
        coverage = 1. - float(nodata_pixels/total_pixels)
        return coverage > self.coverage_floor


    def write_tfrecords(self, tile=None, year=None, dayofyear=None, solar=False):
        file_df = self.file_pairs(tile, year, dayofyear)
        for i in list(range(file_df.shape[0])):
            row = file_df.ix[i]

            # read in preprocessed and normalized data
            name = '{}_{}_{}_{}_{}'.format(row.tile, row.year, row.dayofyear, row.hour, row.minute)

            AHI05_data, AHI12_data, mask = self.load_files(row['ahi05_file'], row['ahi12_file'], solar=solar)

            tile_coverage = np.nanmean(mask)

            # make patches
            AHI05_patches = make_patches(AHI05_data, self.patch_size, self.patch_stride)
            AHI12_patches = make_patches(AHI12_data, self.patch_size, self.patch_stride)
            mask_patches = make_patches(mask, self.patch_size, self.patch_stride)

            record_filename = os.path.join(self.tf_data_dir, '{}.tfrecords'.format(name))
            print("File: {}".format(record_filename))
            writer = tf.io.TFRecordWriter(record_filename)
            written_counter = 0
            for p in list(range(len(AHI05_patches))):
                if self.check_coverage(mask[p]):
                    features = dict()
                    features['AHI05'] = _bytes_feature(AHI05_patches[p].astype(np.float32).tostring())
                    features['AHI12'] = _bytes_feature(AHI12_patches[p].astype(np.float32).tostring())
                    features['mask'] = _bytes_feature(mask_patches[p].astype(np.float32).tostring())
                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())
                    written_counter += 1
            if written_counter == 0:
                os.remove(record_filename)
            print("number of patches written={}".format(written_counter))

    def _tf_decoder(self, serialized_example):
        feature_dict = {'AHI05': tf.io.FixedLenFeature([], tf.string),
                        'AHI12': tf.io.FixedLenFeature([], tf.string),
                        'mask': tf.io.FixedLenFeature([], tf.string)}
        features = tf.io.parse_single_example(serialized_example, features=feature_dict)

        ahi05 = tf.io.decode_raw(features['AHI05'], tf.float32)
        ahi05 = tf.reshape(ahi05, [self.patch_size, self.patch_size, len(self.input_bands)])

        ahi12 = tf.io.decode_raw(features['AHI12'], tf.float32)
        ahi12= tf.reshape(ahi12, [self.patch_size, self.patch_size, 6])

        mask = tf.io.decode_raw(features['mask'], tf.float32)
        mask= tf.reshape(mask, [self.patch_size, self.patch_size, 1])
        return dict(AHI05=ahi05, AHI12=ahi12, mask=mask)

    def make_dataset(self, filenames=None, is_training=True, batch_size=20, buffer_size=100, epochs=None,
                      train_ratio=0.8):
        with tf.name_scope("read_records"), tf.device("/cpu:0"):
            if filenames is None:
                filenames = [os.path.join(self.tf_data_dir, f) for f in os.listdir(self.tf_data_dir)]
                idx = int(len(filenames)*train_ratio)
                if is_training:
                    filenames = filenames[:idx]
                else:
                    filenames = filenames[idx:]

            _parser = lambda x: self._tf_decoder(x)
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(_parser, num_parallel_calls=4)
            dataset = dataset.repeat(epochs)
            dataset = dataset.shuffle(buffer_size=buffer_size)
            dataset = dataset.batch(batch_size)
            dataset = dataset.make_one_shot_iterator()
            element = dataset.get_next()
            return element['AHI05'], element['AHI12'], element['mask']


def create_rsync_nas_file(write_file):
    days = np.arange(1,366,7)
    year = 2017

    nas = NASHimawari()
    tiles = sorted(nas.AHI12_tiles())
    lines = 0
    with open(write_file, 'w') as writer:
        for doy in days:
            for t in tiles:
                print('dayofyear', doy, 'tile', t)
                df1 = nas.AHI05_files(year=year, tile=t, dayofyear=doy)
                df2 = nas.AHI12_files(year=year, tile=t, dayofyear=doy)
                mergedf = df1.merge(df2, left_on=['hour', 'minute'], right_on=['hour', 'minute'])
                for i, row in mergedf.iterrows():
                    if row.minute % 30 == 0:
                        f = row['file_x']
                        relative_file = '/'.join(f.split('/')[6:])
                        writer.write(relative_file + '\n')

                        f = row['file_y']
                        relative_file = '/'.join(f.split('/')[6:])
                        writer.write(relative_file + '\n')
                        lines += 2

    print('number of lines', lines)


### SAVE PREDICTED EXAMPLE TO HDF FILE  #######
def save_prediction(write_file, loc, sigma, mask_probs=None):
    '''['sur_refl_500m', 'RelAZ', 'Fv', 'cosVZA', 'Status_QA', 'Fg', 'sur_refl', 'Snow_Fraction', 'VAZ',
    'SAZ', 'Snow_Grain_Diameter', 'cosSZA', 'Snow_Fit', 'Scattering_Angle', 'Sigma_BRFn', 'Glint_Angle']
    sur_refl:  {'_FillValue': -28672L, 'scale_factor': 0.0001, 'add_offset': 0.0, 'long_name':
        'Surface_bidirectinal_reflectance_factor_1km', 'data description': 'none', 'band_names':
            '1,2,3,4,5,6,7,8,9,10,11,12', 'valid_range': [-100L, 16000L], 'unit': 'reflectance'}
    {'Bands_1KM:grid1km': 12, 'YDim:grid1km': 600, 'XDim:grid1km': 600}
    '''
    def reprocess(x):
        x = x * 1e4
        x = x.astype(np.int16)
        x = np.transpose(x, (2,0,1))
        return x

    loc = reprocess(loc)
    sigma = reprocess(sigma)

    sd = SD(write_file, SDC.WRITE|SDC.CREATE)
    sds_loc = sd.create('sur_refl_mean', SDC.INT16, loc.shape)
    sds_loc[:] = loc
    sds_loc.endaccess()

    sds_sigma = sd.create('sur_refl_std', SDC.INT16, sigma.shape)
    sds_sigma[:] = sigma
    sds_sigma.endaccess()

    if mask_probs is not None:
        mask_probs = reprocess(mask_probs)
        sds_mask= sd.create('mask_probs', SDC.INT16, mask_probs.shape)
        sds_mask[:] = mask_probs
        sds_mask.endaccess()
    sd.end()

def load_prediction(file, mask_probs=False):
    fp = SD(file, SDC.READ)
    loc_obj = fp.select('sur_refl_mean')
    loc = loc_obj.get()[:] * 1e-4
    loc = np.transpose(loc, (1,2,0))
    std_obj = fp.select('sur_refl_std')
    std = std_obj.get()[:] * 1e-4
    std = np.transpose(std, (1,2,0))
    if mask_probs:
        mask_obj = fp.select('mask_probs')
        mask_probs = mask_obj.get()[:]
        return loc, std, mask_probs
    return loc, std


### TEST READING EMULATOR DATA ####
def test_reader():
    em = EmulatorData()
    em.write_tfrecords()
    x, y, m = em.read_tfrecords()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    sess.run(init)
    sess.run([x,y,m])


### TESTING BAND PREDICTION DATA ####
def test_band_prediction_data():
    data = BandData()
    ds = data.tf_dataset_patches()
    sess = tf.Session()
    x, y = sess.run(ds)
    print('input', x.shape, 'label', y.shape)


if __name__ == '__main__':
    em = EmulatorData()
    em.write_tfrecords(tile='h08v13', year=2016, solar=True)
