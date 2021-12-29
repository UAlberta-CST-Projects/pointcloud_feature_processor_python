import numpy as np
import pandas as pd
import pylas
import h5py
from scipy.spatial import cKDTree
from tqdm import tqdm
from os import path, listdir, makedirs
from more_itertools import always_iterable
from typing import Iterable, TYPE_CHECKING

from util import connected_components

if TYPE_CHECKING:
    pass


def from_las(filename: str):
    return PointCloud.from_las(filename)


class PointCloud:
    def __init__(self, data: pd.DataFrame, *, extent_min=None, extent_max=None):
        self._data = data  # type: pd.DataFrame
        self._extent_min = np.array(extent_min if extent_min is not None else (self.x.min(), self.y.min(), self.z.min()))
        self._extent_max = np.array(extent_max if extent_max is not None else (self.x.max(), self.y.max(), self.z.max()))

    @classmethod
    def from_hdf5(cls, filename: str, *, x='x', y='y', z='z', classification='classification'):
        with h5py.File(filename, 'r') as file:
            data = {}
            if x is not None:
                data['x'] = file[x][:].flatten()
            if y is not None:
                data['y'] = file[y][:].flatten()
            if z is not None:
                data['z'] = file[z][:].flatten()
            if classification is not None:
                data['classification'] = file.get(classification, default=np.zeros_like(data['x']))[:].flatten()

        return cls(pd.DataFrame(data))

    def blockwise_uniform_downsample(self, cell_size, agg='intensity', keep_max=True):
        new_data = self.data().assign(
            group_x=(self.x / cell_size).round(),
            group_y=(self.y / cell_size).round(),
            group_z=(self.z / cell_size).round(),
        )
        new_data.sort_values(agg, inplace=True)
        new_data.drop_duplicates(['group_x', 'group_y', 'group_z'], keep='last' if keep_max else 'first', inplace=True)
        new_data.drop(columns=['group_x', 'group_y', 'group_z'], inplace=True)
        return PointCloud(new_data, extent_min=self._extent_min, extent_max=self._extent_max)

    def voxelize(self, cell_size, agg='intensity', keep_max=True):
        new_data = self.data().assign(
            x=(self.x / cell_size).round(),
            y=(self.y / cell_size).round(),
            z=(self.z / cell_size).round(),
        )
        new_data.sort_values(agg, inplace=True)
        new_data.drop_duplicates(['x', 'y', 'z'], keep='last' if keep_max else 'first', inplace=True)
        return PointCloud(new_data, extent_min=self._extent_min, extent_max=self._extent_max)

    def trajectory(self, width=None, eps=0.01, anchor_points=None):
        if anchor_points is None:
            trajectory_mask = np.abs(self.data().scan_angle_rank) < eps
            if width is None:
                return PointCloud(self.data()[trajectory_mask], extent_min=self._extent_min, extent_max=self._extent_max)
            trajectory_tree = cKDTree(self.xyz[trajectory_mask], compact_nodes=False, balanced_tree=False)
        else:
            trajectory_tree = cKDTree(anchor_points.xyz, compact_nodes=False, balanced_tree=False)

        distances, _ = trajectory_tree.query(self.xyz, k=1, distance_upper_bound=width or eps)
        near_points = self.data()[distances < np.inf]
        return PointCloud(near_points, extent_min=self._extent_min, extent_max=self._extent_max)

    @classmethod
    def from_las(cls, filename: str):
        las = pylas.read(filename)
        fields = [f for f in las.points_data.dimensions_names if f not in ['X', 'Y', 'Z', ]] + ['x', 'y', 'z']
        return cls(pd.DataFrame({field: getattr(las, field) for field in fields}))

    @classmethod
    def from_dir(cls, dirname: str, file_type='hdf5', **kwargs):
        for f in listdir(dirname):
            if f.endswith('.h5') and file_type == 'hdf5':
                yield PointCloud.from_hdf5(path.join(dirname, f), **kwargs)
            if f.endswith('.las') and file_type == 'las':
                yield PointCloud.from_las(path.join(dirname, f))

    @classmethod
    def join(cls, pointclouds: 'Iterable[PointCloud]'):
        return PointCloud(pd.concat([pc._data for pc in pointclouds], ignore_index=True, sort=False))

    def merge(self, merge_from: 'PointCloud', *, keep=None, replace=None, eps=0.1, extend=True):
        if keep is None and replace is None:
            replace_cols = set(self._data)
        elif keep is not None:
            replace_cols = set(merge_from._data) - set(always_iterable(keep))
        else:
            replace_cols = set(always_iterable(replace))

        tree = cKDTree(self.xyz, balanced_tree=False, compact_nodes=False)
        _, nn = tree.query(merge_from.xyz, distance_upper_bound=eps, k=1)

        to_replace_idx = self._data.iloc[nn[nn != tree.n]].index
        from_replace_filter = nn != tree.n
        from_add_filter = nn == tree.n

        for col in replace_cols:
            self._data.loc[to_replace_idx, col] = merge_from._data.loc[from_replace_filter, col].values

        if extend:
            self._data = pd.concat((self._data, merge_from._data[from_add_filter]), ignore_index=True, sort=False)
            self._extent_min = np.minimum(self._extent_min, merge_from._extent_min)
            self._extent_max = np.maximum(self._extent_min, merge_from._extent_max)

    @property
    def n_points(self) -> int:
        return self.x.size

    @property
    def xyz(self) -> pd.DataFrame:
        return self._data[['x', 'y', 'z']]

    @xyz.setter
    def xyz(self, new_xyz):
        self._data.loc[:, ['x', 'y', 'z']] = new_xyz

    @property
    def normalized_xyz(self) -> pd.DataFrame:
        return (self.xyz - self._extent_min) / (self._extent_max - self._extent_min) * 2 - 1

    @property
    def x(self) -> pd.Series:
        return self._data['x']

    @property
    def normalized_x(self) -> pd.Series:
        return (self.x - self._extent_min[0]) / (self._extent_max[0] - self._extent_min[0]) * 2 - 1

    @property
    def y(self) -> pd.Series:
        return self._data['y']

    @property
    def normalized_y(self) -> pd.Series:
        return (self.y - self._extent_min[1]) / (self._extent_max[1] - self._extent_min[1]) * 2 - 1

    @property
    def z(self) -> pd.Series:
        return self._data['z']

    @property
    def normalized_z(self) -> pd.Series:
        return (self.z - self._extent_min[2]) / (self._extent_max[2] - self._extent_min[2]) * 2 - 1

    @property
    def classification(self) -> pd.Series:
        return self._data.classification

    @classification.setter
    def classification(self, new_classification):
        self._data.classification = new_classification

    def filter(self, filter_):
        if isinstance(filter_, str):
            return PointCloud(self._data.query(filter_))
        else:
            return PointCloud(self._data.loc[filter_].copy())

    def dropna(self, inplace=False, **kwargs):
        if inplace:
            self._data.dropna(inplace=inplace, **kwargs)
        else:
            return PointCloud(self._data.dropna(inplace=inplace, **kwargs), extent_min=self._extent_min,
                              extent_max=self._extent_max)

    def add_fields(self, **fields):
        self._data = self._data.assign(**{k: np.array(v) for k, v in fields.items()})

    def _split_old(self, step_size: float, *, verbose=False) -> 'Iterable[PointCloud]':
        tree = cKDTree(self.xyz)

        for x in tqdm(np.arange(self._extent_min[0], self._extent_max[0] + step_size, step_size), disable=not verbose):
            for y in np.arange(self._extent_min[1], self._extent_max[1] + step_size, step_size):
                for z in np.arange(self._extent_min[2], self._extent_max[2] + step_size, step_size):
                    ball_indices = tree.query_ball_point((x + step_size/2, y + step_size/2, z + step_size/2), step_size)
                    if len(ball_indices) == 0:
                        continue
                    
                    extent_min = (x, y, z)
                    extent_max = (x + step_size, y + step_size, z + step_size)
                    data_in_ball = self._data.iloc[ball_indices, :]
                    extent_filter = ((extent_min[0] <= data_in_ball.x) & (data_in_ball.x <= extent_max[0]) &
                                     (extent_min[1] <= data_in_ball.y) & (data_in_ball.y <= extent_max[1]) &
                                     (extent_min[2] <= data_in_ball.z) & (data_in_ball.z <= extent_max[2]))
                    data_in_extent = data_in_ball[extent_filter]
                    if data_in_extent.shape[0] == 0:
                        continue

                    # print(extent_min, extent_max)
                    yield PointCloud(data_in_extent, extent_min=extent_min, extent_max=extent_max)

    def split(self, cell_size: dict) -> 'Iterable[PointCloud]':
        pc_data = self.data().assign(
            group_x=self.x // cell_size['x'],
            group_y=self.y // cell_size['y'],
            group_z=self.z // cell_size['z'],
        )
        for group_xyz, group in pc_data.groupby(by=['group_x', 'group_y', 'group_z'], as_index=False):
            extent_min = np.multiply(np.array(group_xyz), np.array([cell_size['x'], cell_size['y'], cell_size['z']]))
            extent_max = np.add(extent_min, np.array([cell_size['x'], cell_size['y'], cell_size['z']]))
            # assert (extent_min <= group[['x', 'y', 'z']].values).all()
            # assert (extent_max >= group[['x', 'y', 'z']].values).all()
            group = group.drop(columns=['group_x', 'group_y', 'group_z'])
            yield PointCloud(data=group, extent_min=extent_min, extent_max=extent_max)

    def sample(self, sample_size: int, *, method='uniform') -> 'PointCloud':
        if method == 'uniform':
            try:
                return PointCloud(self._data.sample(n=sample_size, replace=True), extent_min=self._extent_min, extent_max=self._extent_max)
            except ValueError:
                import pandas as pd
                return PointCloud(pd.DataFrame(dict(x=[0], y=[0], z=[0], intensity=[0], classification=[0])))
        elif method == 'farthest-point':
            if sample_size >= self.n_points:
                return self.sample(sample_size, method='uniform')

            def calc_distances(p):
                return np.array(((p[['x', 'y', 'z']] - self.xyz) ** 2).sum(axis=1).values)

            sample = pd.DataFrame(0, index=np.arange(sample_size), columns=list(self._data))
            sample.iloc[0] = self._data.sample(n=1).iloc[0]
            distances = calc_distances(sample.loc[0])
            for i in range(1, sample_size):
                sample.iloc[i] = self.data().iloc[np.argmax(distances)]
                distances = np.minimum(distances, calc_distances(sample.iloc[i]))

            return PointCloud(sample, extent_min=self._extent_min, extent_max=self._extent_max)
        else:
            raise ValueError('PointCloud.sample() received unknown method \'{}\'. Legal values are \'uniform\' and \'farthest-point\'.'.format(method))

    def data(self):
        return self._data

    def to_hdf5(self, filename: str, *, x='x', y='y', z='z', classification='classification', normalize_xyz=False):
        makedirs(path.dirname(filename), exist_ok=True)
        with h5py.File(filename, 'w') as file:
            if 'x' in self._data:
                file.create_dataset(x, data=self.normalized_x if normalize_xyz else self.x)
            if 'y' in self._data:
                file.create_dataset(y, data=self.normalized_y if normalize_xyz else self.y)
            if 'z' in self._data:
                file.create_dataset(z, data=self.normalized_z if normalize_xyz else self.z)
            if 'classification' in self._data:
                file.create_dataset(classification, data=self.classification)

    def to_las(self, filename: str, *, normalize_xyz=False):
        if path.dirname(filename):
            makedirs(path.dirname(filename), exist_ok=True)
        las = pylas.create(point_format_id=1)
        las.x = self.normalized_x if normalize_xyz else self.x
        las.y = self.normalized_y if normalize_xyz else self.y
        las.z = self.normalized_z if normalize_xyz else self.z
        core_fields = []
        extra_fields = []
        for field in set(self._data) - {'x', 'y', 'z'}:
            try:
                _ = las[field]
                core_fields.append(field)
            except ValueError:
                extra_fields.append(field)

        for field in extra_fields:
            las.add_extra_dim(field, 'f8')
            las[field] = self._data[field].values.astype(las[field].dtype, copy=False)

        for field in core_fields:
            las[field] = self._data[field].values.astype(las[field].dtype, copy=False)

        las.write(filename)
    
    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is not None:
            self_contained_plot = False
        else:
            ax = plt.subplot(projection='3d')
            self_contained_plot = True
        
        ax.plot(self.x, self.y, '.', zs=self.z, **kwargs)
        
        if self_contained_plot:
            plt.show()

    def dropna(self, inplace=False, **kwargs):
        if inplace:
            self._data.dropna(inplace=inplace, **kwargs)
        else:
            return PointCloud(self._data.dropna(inplace=inplace, **kwargs), extent_min=self._extent_min,
                              extent_max=self._extent_max)

    def euclidean_clusters(self, min_distance: float) -> 'List[PointCloud]':
        idx = connected_components.euclidean_cluster_extraction(self.xyz.values, min_distance)
        return [self.filter(idx == i) for i in np.unique(idx)]

if __name__ == '__main__':
    from config import config

    pc = PointCloud.from_las(path.join(config.test_input_dir, '28A03S_C1L1_L1L2_04000_00000.las'))
    pc.to_las(path.join(config.test_input_dir, 'test.las'))
