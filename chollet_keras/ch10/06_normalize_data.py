mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean # shift to mean
stdev = raw_data[:num_train_samples].std(axis=0)
raw_data /= stdev # scale with stdev

# TODO: Check the data mean and stdev now