
import numpy as np

def thresh(x):
    if x < 0:
        return -1
    
    return 1

def main():
    X = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=np.float)

    y = np.array([-1, 1, 1, -1], dtype=np.float)

    W = np.zeros_like(y)

    print(X)
    print(y)

    num_epochs = 2
    learning_rate = 0.2

    for epoch in range(num_epochs):
        for sample in range(len(X)):
            print("epoch_" + str(epoch) + "_sample_" + str(sample))

            predicted_val = thresh(np.dot(X[sample], W))
            print("predicted val: " + str(predicted_val))

            change_W = learning_rate * (y[sample] - predicted_val) * X[sample]
            print("change_W: " + str(change_W))

            W += change_W
            print("new_W: " + str(W))

if __name__ == "__main__":
    main()
