
import skimage
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage import data, segmentation, graph

# 1. Load the coffee image
coffee = data.coffee()

start = time.time()

# 2. Divide the image into superpixels using SLIC
slic = segmentation.slic(
    coffee,
    compactness=20,
    n_segments=600,
    start_label=1,
    channel_axis=-1
)

# 3. Create a region adjacency graph
g = graph.rag_mean_color(
    coffee,
    slic,
    mode='similarity'
)

# 4. Apply the Normalized Cut algorithm
ncut = graph.cut_normalized(slic, g)

# 5. Print processing time
print(
    coffee.shape,
    'Coffee 영상을 분할하는 데',
    time.time() - start,
    '초 소요'
)

# 6. Draw the segmentation boundaries
marking = segmentation.mark_boundaries(coffee, ncut)

# 7. Display the result in Spyder
plt.figure(figsize=(10, 6))
plt.imshow(marking)
plt.title('Normalized Cut')
plt.axis('off')
plt.tight_layout()
plt.show()
