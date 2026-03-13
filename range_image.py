import numpy as np
def spherical_projection(xyz, H=64, W=2048, f_down=-25, f_up=3.0):
    x, y, z, intensity = xyz['x'].values, xyz['y'].values, xyz['z'].values, xyz['intensity'].values
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    f_down = f_down / 180.0 * np.pi
    f_up = f_up / 180.0 * np.pi
    f = abs(f_down) + abs(f_up)

    u = 0.5 * (1 - np.arctan2(y, x) / np.pi) * W
    v = (1 - (np.arcsin(z / r) + abs(f_down)) / f) * H

    u = np.floor(u).astype(int)
    v = np.floor(v).astype(int)

    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    return u, v, r


def range_image(xyz, H=64, W=2048):
    x, y, z, intensity = xyz['x'].values, xyz['y'].values, xyz['z'].values, xyz['intensity'].values
    u, v, r = spherical_projection(xyz, H=H, W=W, f_down=-25, f_up=3.0)
    range_image = np.zeros((5, H, W), dtype=np.float32)

    for i in range(len(x)):
        if range_image[0, v[i], u[i]] == 0 or r[i] < range_image[0, v[i], u[i]]:
            range_image[0, v[i], u[i]] = r[i]
            range_image[1, v[i], u[i]] = x[i]
            range_image[2, v[i], u[i]] = y[i]
            range_image[3, v[i], u[i]] = z[i]
            range_image[4, v[i], u[i]] = intensity[i]

    return range_image