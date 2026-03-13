import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def visualization_2D(lidar_df,color=None):
    if color is not None: color=np.asarray(color, dtype=np.float32)
    else: color=lidar_df['intensity']
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(-lidar_df['x'], -lidar_df['y'], c=color, s=1)
    plt.grid()
    plt.xlabel('x, m', fontsize=26)
    plt.ylabel('y, m', fontsize=26)
    plt.title('LiDAR', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.show()

def visualization_3D(xyz,labels=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    if labels is not None: color=np.asarray(labels, dtype=np.float32)
    else: color=xyz[:, 2]
    fig = go.Figure(data=[go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=color,
            colorscale='Turbo'
        )
    )])

    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig.show()