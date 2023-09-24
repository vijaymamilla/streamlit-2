
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)
@st.cache_data
def loadTrainDf():
    train_df = pd.read_csv('app/artifactory/lome_train1.csv', index_col='date_time', parse_dates=True)

    return train_df
@st.cache_data
def loadValDf():
    val_df = pd.read_csv('app/artifactory/lome_val1.csv', index_col='date_time', parse_dates=True)
    return val_df
@st.cache_data
def loadTestDf():
    test_df = pd.read_csv('app/artifactory/lome_test1.csv', index_col='date_time', parse_dates=True)

    return test_df


class DataWindow():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)


@st.cache_resource
def buildLstmModel():

    new_model = keras.models.load_model("app/artifactory/lstm_model.h5")

    return new_model


def buildDataFrame(days:int):
    custom_mo_wide_window = DataWindow(input_width=days, label_width=days, shift=days,train_df=loadTrainDf(), val_df=loadValDf(), test_df=loadTestDf(),
                                       label_columns=['precipitation_sum (mm)', 'rain_sum (mm)', 'river_discharge',
                                                      'intensity_rain', 'intensity_flood', 'intensity_drought'])
    #input_indices = custom_mo_wide_window.input_indices
    #label_indices = custom_mo_wide_window.label_indices

    predicted_results = buildLstmModel().predict(custom_mo_wide_window.test)
    predicted_array = predicted_results[0]

    my_array = np.array(predicted_array)

    df = pd.DataFrame(my_array)

    df2 = df.rename(columns={0: "precipitation_sum (mm)", 1: "rain_sum (mm)", 2: "river_discharge", 3: "intensity_rain",4: "intensity_flood", 5: "intensity_drought"})

    st.write("14 days forecast")
    st.dataframe(df2)


buildDataFrame(14)