from dataset import download_data, get_pairs, ohlc_to_state_value

if __name__ == '__main__':
    # print(get_pairs())
    # quit()
    import time
    from tensorflow.keras.models import load_model
    model = load_model('model/net_50')
    true, false = 0, 0

    data_size = 16
    q = 8
    for pair in get_pairs():
        print(pair)
        x = download_data(pair=pair, path='', save=False, since=int(time.time()) - (data_size * q + 1) * 60 * 60)
        for i in range(1, q):
            idx = i * data_size 
            s = ohlc_to_state_value(x[-1-idx-data_size:-1-idx], data_size, False, True)
            if s is None:
              continue
            y = model.predict(s)[0][0]
            diff = float(x[-2-idx][4])-float(x[-1-idx][4])
            if (y < 0 and diff < 0) or (y > 0 and diff > 0):
                true += 1
            else:
                false += 1
        print(true, false)