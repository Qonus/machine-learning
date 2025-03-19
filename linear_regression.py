import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
import random

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

points = []
for i in range(1, 10):
    t = random.randrange(0, 10)
    points.append((t, t + random.randrange(-5, 5)))
points = np.array(points)
baby_step = 0.01
steps = 100
w = 0
b = 0

loss_graph = []
w_values = [w]
b_values = [b]

for i in range(1, steps):
    loss = calculate_loss(points, w, b)
    loss_graph.append(loss)
    slope = calculate_slope(points, w, b)
    w -= slope[0] * baby_step
    b -= slope[1] * baby_step
    w_values.append(w)
    b_values.append(b)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))
fig.subplots_adjust(bottom=0.25)

xloss = np.arange(1, steps)
yloss = np.array(loss_graph)

l1, = ax1.plot(xloss, loss_graph)
ax1.set_xlabel('Generation')
ax1.set_ylabel('Loss')

xpoints = points[:, 0]
ypoints = points[:, 1]

ax2.plot(xpoints, ypoints, "o")

x = np.linspace(0, 10, 100)
line, = ax2.plot(x, w_values[0] * x + b_values[0])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# for i in range(len(xpoints)):
#     plt.plot([xpoints[i], xpoints[i]], [ypoints[i], w_values[0] * points[i,0] + b_values[0]], color='gray', linestyle='--')

axgen = fig.add_axes([0.25, 0.1, 0.65, 0.03])
gen = Slider(ax=axgen,label='Generation', valmin=0, valmax=steps - 1, valinit=0, valstep=1)

def update_gen(val):
    val = int(val)
    l1.set_ydata(loss_graph[:val])
    l1.set_xdata(xloss[:val])
    line.set_ydata(w_values[val] * x + b_values[val])
    ax1.set_xlim(1,val)
    fig.canvas.draw_idle()

gen.on_changed(update_gen)

nextax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(nextax, 'Next', hovercolor='0.975')

def next_gen(event):
    gen.set_val(gen.val + 1)
button.on_clicked(next_gen)

plt.show()