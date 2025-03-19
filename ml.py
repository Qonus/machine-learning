import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

def calculate_loss(points, w, b) -> int:
    loss = 0
    for point in points:
        prediction = w * point[0] + b
        loss += np.square(prediction - point[1])
    return loss

def calculate_slope(points, w, b) -> list[int]:
    dw = 0
    db = 0
    M = len(points)
    for point in points:
        prediction = w * point[0] + b
        dw += (prediction - point[1]) * 2 * point[0]
        db += (prediction - point[1]) * 2
    return [dw / M, db / M]

points = np.array([(1, 4), (3, 4), (5, 5)])
baby_step = 0.01
steps = 1000
w = 0
b = 0

loss_graph = []

for i in range(1, steps):
    loss = calculate_loss(points, w, b)
    loss_graph.append(loss)
    slope = calculate_slope(points, w, b)
    w -= slope[0] * baby_step
    b -= slope[1] * baby_step

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))

xloss = np.arange(1, steps)
yloss = np.array(loss_graph)

l1, = ax1.plot(xloss, loss_graph)

xpoints = points[:, 0]
ypoints = points[:, 1]

ax2.plot(xpoints, ypoints, "o")

x = np.linspace(0, 10, 100)
ax2.plot(x, w*x + b)


axgen = plt.axes([0.25, 0.15, 0.65, 0.03])
gen = Slider(axgen, 'Generation', 1, steps - 1, 1, valstep=1)

def update_gen(val):
    #l1.set_ydata(val)
    print(val)

gen.on_changed(update_gen)

plt.show()