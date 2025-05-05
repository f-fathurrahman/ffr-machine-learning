# Computing number of samples we will use for each data split
num_train_samples = int(0.5*len(raw_data))
num_val_samples = int(0.25*len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples

print("num_train_samples = ", num_train_samples)
print("num_val_samples = ", num_val_samples)
print("num_test_samples = ", num_test_samples)

