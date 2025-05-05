import matplotlib.pyplot as plt

plt.clf()
plt.plot(range(len(temperature)), temperature)
plt.grid(True)
plt.xlabel("Timestamp")
plt.ylabel("Temperature (deg Celcius)")
plt.savefig("IMG_03_plot_temperature.png", dpi=150)

