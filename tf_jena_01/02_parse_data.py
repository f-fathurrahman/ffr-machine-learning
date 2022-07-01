import numpy as np

temperature = np.zeros( (len(lines),) )
raw_data = np.zeros( (len(lines), len(header)-1) )

for i, line in enumerate(lines):
    values = [ float(x) for x in line.split(",")[1:] ]
    # We store column 1 in the "temperature" array
    temperature[i] = values[1]
    # We store all columns (including the temperature) in the raw_data array
    raw_data[i,:] = values[:]

# exec(open(filename).read())

