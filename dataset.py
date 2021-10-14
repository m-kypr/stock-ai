from os import stat
import requests as rq
import time
import json
import numpy as np
from requests.api import get

def download_data(pair, path, save=True, since=1):
    interval = 60 
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}&since={since}"
    r = rq.get(url)
    jj = json.loads(r.text)
    data = jj['result'][list(jj['result'].keys())[0]]
    last = jj['result']['last']
    since = int(last)
    if save:
        open(path, 'w').write(json.dumps(data))
    return data

def get_pairs(curr='usd'):
    url = "https://api.kraken.com/0/public/AssetPairs"
    r = rq.get(url) 
    return [x for x in list(json.loads(r.text)['result'].keys()) if x[-3:].lower() == curr.lower()]

def download_all():
    for pair in get_pairs():
        download_data(pair, f'data/{pair}.json')

def ohlc_to_state_value(ohlc, data_size, get_value=True, expand_dims=False):
    # f(X) = Y 
    # [int <time>, string <open>, string <high>, string <low>, string <close>, string <vwap>, string <volume>, int <count>]
    
    points = [float(x[4]) for x in ohlc]
    if get_value:
        mn = min(points[:-1])
    else:
        mn = min(points)
    scaled_points = [x - mn for x in points]
    if get_value:
        mx = max(scaled_points[:-1])
    else:
        mx = max(scaled_points)
    if mx == 0:
        if get_value:
            return None, None
        else:
            return None
    normalized_points = [x / mx for x in scaled_points]
    
    if get_value:
        pointsX = normalized_points[:-1]
        values = []
        for y in normalized_points[-1:]:
            if y > 1:
                values.append(1)
            elif y < 0:
                values.append(-1)
            else:
                values.append(0)
    else: 
        pointsX = normalized_points
    
    volumes = [float(x[6]) for x in ohlc]
    if get_value:
        mn = min(volumes[:-1])
    else:
        mn = min(volumes)
    volumes = [x - mn for x in volumes]
    if get_value:
        mx = max(volumes[:-1])
    else:
        mx = max(volumes)
    normalized_volumes = [x / mx for x in volumes]
    if get_value:
        volumesX = normalized_volumes[:-1]
    else:
        volumesX = normalized_volumes
    
    if get_value:
        state = np.zeros((2,data_size-1), np.float)
        state[0] = pointsX
        state[1] = volumesX
        return state, values
    else:
        state = np.zeros((2,data_size), np.float)
        state[0] = pointsX
        state[1] = volumesX
        if expand_dims:
            return np.expand_dims(state, axis=0)
        return state




def get_dataset(n):
    # [int <time>, string <open>, string <high>, string <low>, string <close>, string <vwap>, string <volume>, int <count>]
    data_size = 16 + 1
    X, Y = [], []
    import os
    for fn in os.listdir('data'):
        ohlc = json.loads(open(os.path.join('data', fn), 'r').read())
        l = len(ohlc) # 720
        x = int(l / data_size) # 11
        for i in range(0, x): 
            idx = i * data_size
            s, v = ohlc_to_state_value(ohlc[idx:idx+data_size], data_size)
            if s is not None:
                X.append(s)
                Y.append(v)
        print(f'\rgot #{len(X)} examples', end='')
        if len(X) > n:
            break
        # quit()

    # X = np.array(X, ndmin=2)
    # Y = np.array(Y)
    print()
    return X, Y 


def get_kraken(n):
    path = 'E:\Datasets\stock\Kraken'
    data_size = 32 + 1
    import os
    import csv
    X, Y = [], []
    for fn in os.listdir(path): 
        sp = fn.split('_')
        if sp[1] == '60.csv':
            with open(os.path.join(path, fn), 'r', newline='') as f:
                ohlc = list(csv.reader(f))
                l = len(ohlc)
                x = int(l / data_size)
                for i in range(0, x): 
                    idx = i * data_size
                    s, v = ohlc_to_state_value(ohlc[idx:idx+data_size], data_size)
                    if s is not None:
                        X.append(s)
                        Y.append(v)
                print(f'\rgot #{len(X)} examples', end='')
                if len(X) > n:
                    break
    return X, Y




if __name__ == '__main__':
    X, Y = get_kraken(250000)
    # X, Y = get_dataset(3000)
    np.savez("processed/dataset_250K.npz", X, Y)