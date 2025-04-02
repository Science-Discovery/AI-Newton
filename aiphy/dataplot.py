import matplotlib.pyplot as plt
from .interface import DataStruct, NormalData, ExpData

def plot_normaldata(data: NormalData, key: str = "<normaldata>"):
    arr = data.data
    for i in range(len(arr)):
        if i == 0:
            plt.plot(arr[i], label = key)
        else:
            plt.plot(arr[i])

def plot_data(data: ExpData, key: str = "<expdata>"):
    if data.is_err:
        print(f'{key}: Error data')
        return
    if data.is_zero:
        print(f'{key}: {data}')
        return
    if data.is_const:
        const_data = data.const_data
        mean = const_data.mean
        std = const_data.std
        # 画一条水平的线
        plt.axhline(y = mean, color = 'r', linestyle = '--')
        # 画一个 error bar
        plt.errorbar(0, mean, yerr = std, fmt = 'o', color = 'r', label = key)
        print(f'{key}: const data, mean = {mean}, std = {std}')
        return
    res: NormalData = data.normal_data
    plot_normaldata(res, key)

def plot_datastruct(data: DataStruct):
    t = data.fetch_data_by_str('t[0]')
    t = t.normal_data.data
    for key in data.data_keys:
        if str(key) == 't[0]':
            continue
        print(f'plot {key}')
        plot_data(data.fetch_data_by_key(key), str(key))
    plt.legend()
