# Stock AI - Can a neural network predict future market moves with past data?

### The Neural Network

I threw some convolutional layers together, input shape is (channels, 2, 16).

16 is the frame size i.e the amount of stock points and volumes in one frame

I train 100 epochs on 250K dataset

### The Dataset

I used Crypto OHLC data, but anything OHLC like will do the job

I scale all the values in a frame to go from 0 to 1 and if the next value (stock price) ...

- ... exceeds the maximum in this frame the label for that frame is 1
- ... is below 0 the label for this frame is -1
- else it is 0

If someone has a better way of labeling, please tell me

### My Evaluation

When evaluating the network on unseen data, we get almost every time a 50/50 split of true and false predictions!

Does that mean stock predicting AIs do not work?

No, but AIs that only consider past stock data, do not work because it is at heart random (brownian) motion.
AIs are currently in use in the wild by big cooperations but those AIs do natrual language processing and therefore take into account news data!

## TODO

- Add more extensive info in README
