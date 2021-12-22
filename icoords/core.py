import re
import types

import xarray as xr

from icoords.interpolate import LinearCoordinate

from .formating_html import complete_html


class DataArrayWrapper(type):

    __ignore__ = ["__class__", "__mro__", "__new__", "__init__",
                  "__setattr__", "__getattr__", "__getattribute__"]

    def __new__(mcs, name, bases, class_dict):
        cls = super().__new__(mcs, name, bases, class_dict)
        names = [name for name in dir(xr.DataArray)
                 if name not in mcs.__ignore__ and name not in class_dict]
        for name in names:
            setattr(cls, name, cls.wrap_attribute(name))
        return cls

    def wrap_attribute(cls, name):
        def fget(self):
            attr = getattr(self.data_array, name)
            if isinstance(attr, types.MethodType):
                return cls.wrap_method(self, attr)
            else:
                return attr

        def fset(self, value):
            return setattr(self.data_array, name, value)

        def fdel(self):
            return delattr(self.data_array, name)

        doc = getattr(xr.DataArray, name).__doc__
        return property(fget, fset, fdel, doc)

    def wrap_method(cls, self, attr):
        def method(*args, **kwargs):
            args = cls.wrap_arguments(self, args)
            result = attr(*args, **kwargs)
            return cls.wrap_result(self, result)
        return method

    def wrap_arguments(cls, self, args):
        wrapped_args = []
        for arg in args:
            if type(arg) is type(self):
                wrapped_args.append(arg.data_array)
            else:
                wrapped_args.append(arg)
        return wrapped_args

    def wrap_result(cls, self, result):
        if compatible(self.data_array, result):
            icoords = adapt_icoords(result, self.icoords)
            return cls(result, icoords)
        else:
            return result


def compatible(data_array, other):
    """Check if other is a subset of self with same subshapes"""
    if not type(other) == type(data_array):
        return False
    if not set(other.dims).issubset(set(data_array.dims)):
        return False
    subsizes = {dim: data_array.sizes[dim] for dim in other.dims}
    if not other.sizes == subsizes:
        return False
    return True


def adapt_icoords(data_array, icoords):
    icoords = {dim: icoords[dim] for dim in data_array.dims}
    return InterpolatedCoordinates(icoords)


class InterpolatedDataArray(metaclass=DataArrayWrapper):

    def __init__(self, data_array, icoords):
        self.data_array = data_array
        self.icoords = icoords

    def __array_ufunc__(self, *args, **kwargs):
        args = InterpolatedDataArray.wrap_arguments(self, args)
        result = self.data_array.__array_ufunc__(*args, **kwargs)
        return InterpolatedDataArray.wrap_result(self, result)

    def __getitem__(self, item):
        data_array = self.data_array[item]
        query = get_query(item, self.icoords.dims)
        dct = {dim: self.icoords[dim][query[dim]]
               for dim in query}
        icoords = InterpolatedCoordinates(dct)
        return InterpolatedDataArray(data_array, icoords)

    @property
    def loc(self):
        return LocIndexer(self)

    def isel(self, **kwargs):
        return self[kwargs]

    def sel(self, **kwargs):
        return self.loc[kwargs]

    def __repr__(self):
        return repr(self.data_array) + "\n" + repr(self.icoords)

    def __str__(self):
        return repr(self)

    def _repr_html_(self):
        html = self.data_array._repr_html_()
        return complete_html(html, self.icoords)

    def load_icoords(self):
        """Interpolate coordinates and load them into memory"""
        for dim in self.dims:
            self.data_array.coords[dim] = (
                self.icoords[dim].values())
        return self.data_array

    def load(self, **kwargs):
        return self.load_icoords().load(**kwargs)

    def compute(self, **kwargs):
        return self.load_icoords().compute(**kwargs)

    @classmethod
    def from_netcdf(cls, *args, **kwargs):
        """Read a netCDF file following the CF conventions about interpolated 
        coordinates"""
        dataset = xr.open_dataset(*args, **kwargs)
        data_array = [var for var in dataset.values()
                      if "coordinate_interpolation" in var.attrs]
        if len(data_array) == 1:
            data_array = data_array[0]
        else:
            ValueError("several possible data arrays detected")
        icoords = InterpolatedCoordinates()
        mapping = data_array.attrs.pop("coordinate_interpolation")
        matches = re.findall(r"(\w+): (\w+) (\w+)", mapping)
        for match in matches:
            dim, indices, values = match
            icoords[dim] = LinearCoordinate(dataset[indices], dataset[values])
        return cls(data_array, icoords)

    def to_netcdf(self, *args, **kwargs):
        """Write a netCDF file using the CF conventions about interpolated
         coordinates"""
        data_arrays = []
        mapping = ""
        for dim in self.icoords.dims:
            mapping += f"{dim}: {dim}_indices {dim}_values "
            interpolation = xr.DataArray(
                name=f"{dim}_interpolation",
                attrs={
                    "interpolation_name": self.icoords[dim].kind,
                    "tie_points_mapping": f"{dim}_points: {dim}_indices {dim}_values",
                }
            ).astype("i4")
            indices = xr.DataArray(
                name=f"{dim}_indices",
                data=self.icoords[dim].tie_indices,
                dims=(f"{dim}_points"),
            )
            values = xr.DataArray(
                name=f"{dim}_values",
                data=self.icoords[dim].tie_values,
                dims=(f"{dim}_points"),
            )
            data_arrays.extend([interpolation, indices, values])
        data_array = self.data_array.copy(deep=False)
        data_array.attrs["coordinate_interpolation"] = mapping
        data_arrays.append(data_array)
        dataset = xr.Dataset({xarr.name: xarr for xarr in data_arrays})
        dataset.to_netcdf(*args, **kwargs)


class InterpolatedCoordinates(dict):
    """Subclass of dict that contains interpolated coordinates for each 
    dimension"""

    @property
    def dims(self):
        return tuple(self.keys())

    @property
    def ndim(self):
        return len(self)

    def __repr__(self):
        s = "Interpolated Coordinates:\n"
        for dim, icoord in self.items():
            s += f"  * {dim}".ljust(12)
            s += f"({dim}) "
            s += repr(icoord) + "\n"
        return s

    def to_index(self, item):
        query = get_query(item, self.dims)
        return {dim: self[dim].to_index(query[dim]) for dim in query}


class LocIndexer:

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, item):
        item = self.obj.icoords.to_index(item)
        return self.obj[item]


def get_query(item, dims):
    query = {dim: slice(None) for dim in dims}
    if isinstance(item, dict):
        query.update(item)
    elif isinstance(item, tuple):
        for k in range(len(item)):
            query[dims[k]] = item[k]
    else:
        query[dims[0]] = item
    return query
