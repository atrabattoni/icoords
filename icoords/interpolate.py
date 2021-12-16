import numpy as np


class LinearCoordinate:
    """Piecewise linear coordinate"""

    def __init__(self, tie_indices, tie_values):
        self.tie_indices = np.asarray(tie_indices)
        self.tie_values = np.asarray(tie_values)
        self.kind = "linear"

    def __repr__(self):
        return (f"{len(self.tie_indices)} tie points from {self.tie_values[0]} "
                f"to {self.tie_values[-1]}")

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.slice(item)
        else:
            return self.get_value(item)

    @property
    def dtype(self):
        return self.tie_values.dtype

    def get_value(self, index):
        return _linear_interpolate(index, self.tie_indices, self.tie_values)

    def get_index(self, value, method="nearest"):
        index = _linear_interpolate(value, self.tie_values, self.tie_indices)
        if method == "nearest":
            return np.rint(index).astype("int")
        elif method == "before":
            return np.floor(index).astype("int")
        elif method == "after":
            return np.ceil(index).astype("int")
        else:
            raise ValueError(
                "valid methods are: 'nearest', 'before', 'after'")

    def indices(self):
        return np.arange(self.tie_indices[-1] + 1)

    def values(self):
        return self.get_value(self.indices())

    def get_index_slice(self, value_slice):
        start = self.get_index(value_slice.start, method="after")
        end = self.get_index(value_slice.stop, method="before")
        stop = end + 1
        return slice(start, stop)

    def slice(self, index_slice):
        start_index = index_slice.start
        end_index = index_slice.stop - 1
        start_value = self.get_value(start_index)
        end_value = self.get_value(end_index)
        mask = (start_index < self.tie_indices) & (
            self.tie_indices < end_index)
        tie_indices = np.insert(
            self.tie_indices[mask],
            (0, self.tie_indices[mask].size),
            (start_index, end_index))
        tie_values = np.insert(
            self.tie_values[mask],
            (0, self.tie_values[mask].size),
            (start_value, end_value))
        tie_indices -= tie_indices[0]
        return LinearCoordinate(tie_indices, tie_values)

    def simplify(self, epsilon):
        """Remove unnecessary tie points using the Ramer-Douglas-Peucker
        algorithm"""
        self.tie_indices, self.tie_values = _simplify(
            self.tie_indices, self.tie_values, epsilon)


class ScaleOffset:
    """Scale-offset transform to ensure the interpolation of datetime data"""

    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    @classmethod
    def floatize(cls, arr):
        if np.issubdtype(arr.dtype, np.datetime64):
            scale = np.timedelta64(1, 'us')
            offset = arr[0]
        else:
            scale = 1.0
            offset = 0.0
        return cls(scale, offset)

    def direct(self, arr):
        return (arr - self.offset) / self.scale

    def inverse(self, arr):
        if np.issubdtype(np.asarray(self.scale).dtype, np.timedelta64):
            arr = np.rint(arr)
        return self.scale * arr + self.offset


def _linear_interpolate(x, xp, fp):
    """Wrapping of numpy interp function to ensure datetime interpolation"""
    if not _is_strictly_increasing(xp):
        raise ValueError("xp must be strictly increasing")
    x_transform = ScaleOffset.floatize(xp)
    f_transform = ScaleOffset.floatize(fp)
    x = x_transform.direct(x)
    xp = x_transform.direct(xp)
    fp = f_transform.direct(fp)
    f = np.interp(x, xp, fp)
    f = f_transform.inverse(f)
    return f


def _is_strictly_increasing(x):
    if np.issubdtype(x.dtype, np.datetime64):
        return np.all(np.diff(x) > np.timedelta64(0, 'us'))
    else:
        return np.all(np.diff(x) > 0)


def _simplify(x, y, epsilon):
    """Ramer-Douglas-Peucker algorithm"""
    mask = np.ones(len(x), dtype=bool)
    stack = [(0, len(x))]
    while stack:
        start, stop = stack.pop()
        ysimple = _linear_interpolate(
            x[start:stop],
            x[[start, stop-1]],
            y[[start, stop-1]],
        )
        d = np.abs(y[start:stop] - ysimple)
        index = np.argmax(d)
        dmax = d[index]
        index += start
        if dmax > epsilon:
            stack.append([start, index+1])
            stack.append([index, stop])
        else:
            mask[start+1:stop-1] = False
    return x[mask], y[mask]
