import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

PATH = "data/"


def read_files(pop):
    name_space = []
    with open(f"{PATH}nsJanelas_{pop}.txt", "r") as file:
        for line in file:
            name_space.append(sorted(list(map(int, line.split()))))

    def read_csv_file(name):
        file = pd.read_csv(
            f"{PATH}/{name}_{pop}.txt", low_memory=False, sep=" ", index_col="NameSpace"
        )
        if file.empty:
            raise FileNotFoundError(f"{PATH}/{name}_{pop}.txt")
        return file

    return read_csv_file("access"), read_csv_file("target"), read_csv_file("vol_bytes")


def filter_dataframe_with_volume(
    dataframe, arq_vol_bytes, filter_window, firstData=None, lastData=None
):
    volume_at_window = arq_vol_bytes.iloc[:, filter_window]
    index_of_files_that_exists = volume_at_window.values.ravel() > 0.0
    df = dataframe[index_of_files_that_exists]
    if firstData is not None:
        df = df.iloc[:, firstData:lastData]
        df.columns = range(df.shape[1])
    df.columns = range(df.shape[1])
    return df


def filter_by_date(dataframe, firstData, lastData):
    if firstData is not None:
        return dataframe.iloc[:,firstData:lastData]
    return dataframe


def filter_dataframe(
    dataframe, arq_vol_bytes, filter_window, firstData=None, lastData=None
):
    df = filter_by_volume(dataframe, arq_vol_bytes, filter_window)
    return filter_by_date(df, firstData, lastData)


def filter_by_volume(dataframe, arq_vol_bytes, filter_window):
    volume_at_window = arq_vol_bytes.iloc[:, filter_window]
    return dataframe[volume_at_window.values.ravel() > 0.0]


def normalize_data(data_series):
    scaler = StandardScaler()
    data_reshaped = data_series.values.reshape(-1, 1)
    return pd.Series(scaler.fit_transform(data_reshaped).ravel())
    # version without scaling
    # return pd.Series(data_reshaped.ravel())


def scale_volume_bytes(arq_vol_bytes):
    scaler = MinMaxScaler()
    return pd.DataFrame(
        scaler.fit_transform(arq_vol_bytes),
        columns=arq_vol_bytes.columns,
        index=arq_vol_bytes.index,
    )


def prepare_training_data(train, trainLabel):
    x_train = train.dropna()
    y_train = trainLabel.dropna()
    y_train = y_train.apply(lambda row: int(row.any()), axis=1)
    # Print how much 0 and 1 we have in the dataset
    # print(y_train.value_counts())

    # if y_train.unique().size > 1:
    #     sm = SMOTE(sampling_strategy="minority")
    #     x_train, y_train = sm.fit_resample(x_train, y_train)
    #     print(y_train.value_counts())

    return x_train, y_train
