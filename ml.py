import matplotlib.pyplot as plt
import numpy as np

points = np.array([(1, 4), (3, 4), (5, 5)])

xpoints = points[:, 0]
ypoints = points[:, 1]

def calculate_loss(points, w, b) -> int:
    loss = 0
    for point in points:
        prediction = w * point[0] + b
        loss += np.square(prediction - point[1])
    return loss

def calculate_slope(points, w, b) -> list[int]:
    
    return w

loss = calculate_loss(points, 0, 0)
print(loss)
plt.plot(xpoints, ypoints, "o")
plt.show()