import torch
import numpy as np
import matplotlib.pyplot as plt

HOME = "01-video-object-detection"

car_count_list = torch.load("outputs/counts/car_count_per_frame.pt")
car_counts = np.array(car_count_list)

for frame in range(0, len(car_counts), 60):
    print(f"Frame {frame}: {car_counts[frame]}")


print("\nDETECTION STATISTICAL SUMMARY")
print("-" * 40)
print(f"Maximum cars detected:      {car_counts.max()}")
print(f"Average cars per frame:     {car_counts.mean():.2f}")
print(f"Median:                     {np.median(car_counts)}")
print(f"Standard deviation (std):   {car_counts.std():.2f}")
print("-" * 40)

plt.plot(car_counts)
plt.title("Car count per frame")
plt.xlabel("Car count")
plt.ylabel("Frames")
plt.savefig("outputs/car_count_per_frame.png")
