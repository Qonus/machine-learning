# import matplotlib
# matplotlib.use('TkAgg')  # 'TkAgg', 'Qt5Agg', 'QtAgg', 'WXAgg'
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.widgets import Slider

# Sample ML data (replace with your model's output)
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
line, = ax.plot(x, y)

axcolor = 'lightgoldenrodyellow'
ax_freq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
freq_slider = Slider(ax_freq, 'Frequency', 0.1, 30.0, valinit=1)

def update(val):
    freq = freq_slider.val
    new_y = np.sin(x * freq)
    line.set_ydata(new_y)
    fig.canvas.draw_idle()

freq_slider.on_changed(update)

plt.show()