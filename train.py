import numpy as np 

def make_dataset(path):
  with np.load(path, allow_pickle=True) as dataset:
      X = dataset['arr_0']
      Y = dataset['arr_1']
      return X, Y


def build_model():
    import tensorflow as tf 
    from tensorflow import keras
    from tensorflow.keras import layers
    with tf.device('/DML:0'):
        model = keras.Sequential()

        ls = [
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape = (2, 16), data_format='channels_first'),
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', strides=2),
            layers.Conv1D(64, kernel_size=2, activation='relu', padding='same'),
            layers.Conv1D(64, kernel_size=2, activation='relu', padding='same'),
            layers.Conv1D(128, kernel_size=2, activation='relu', padding='same', strides=2),
            layers.Conv1D(128, kernel_size=1, activation='relu', padding='same'),
            # layers.BatchNormalization(),
            # layers.Conv2D(32, kernel_size=3, activation='relu',
            #               padding='same'),
            # layers.BatchNormalization(),
            # layers.Dense(64, activation='relu'),
            layers.Flatten(),
            layers.Dense(1, activation='tanh')
        ]
        for i in range(len(ls)):
            l = ls[i]
            model.add(l)
        model.compile(optimizer='adam',
                        loss='mean_squared_error', metrics=['accuracy', 'mse'])
        return model

if __name__ == "__main__":
    from sys import argv
    if len(argv) < 2:
        X, Y = make_dataset('processed/dataset_250K.npz')
        
        from tensorflow.keras.models import load_model
        model = load_model('model/net_50')
        # model = build_model()
        model.summary()
        epochs = 50
        import tensorflow as tf 
        with tf.device('/DML:0'):
            history = model.fit(X, Y, epochs=epochs, validation_split=0.1,
                            shuffle=True, batch_size=256 * 32 * 2)

            np.save('history1.npy', history.history)
            model.save(f'model/net_{epochs}')
    
    else:
        import matplotlib.pyplot as plt
        history=np.load('history1.npy',allow_pickle='TRUE').item()
        print(history.keys())
        plt.plot(history['loss'], label='loss')
        plt.plot(history['acc'], label='accuracy')
        plt.plot(history['val_loss'], label='val_loss')
        plt.plot(history['val_acc'], label='val_accuracy')
        plt.legend()
        plt.ylim([0, 1])
        plt.show()