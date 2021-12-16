import re

import xarray as xr

from icoords.interpolate import LinearCoordinate

from .formating_html import complete_html


class DataArrayWrapper(type):
    """Metaclass that allows to mimic the inheritence of the DataArray class 
    while using composition"""

    __ignore__ = ["__class__", "__mro__", "__new__", "__init__",
                  "__setattr__", "__getattr__", "__getattribute__"]

    def compatible(cls, self, other):
        """Check if other has sames shape and dims"""
        return (
            type(self.data_array) == type(other) and
            self.data_array.dims == other.dims and
            self.data_array.shape == other.shape)

    def wrap(cls, name):
        """Wrap a DataArray method to output InterpolatedDataArray if possible"""

        def method(self, *args, **kwargs):
            if args or kwargs:
                result = getattr(self.data_array, name)(*args, **kwargs)
            else:
                result = getattr(self.data_array, name)
            if cls.compatible(self, result):
                return InterpolatedDataArray(result, self.icoords)
            else:
                return result
        return method

    def __new__(mcs, name, bases, class_dict):
        """Create new instance with wrapped methods of the DataArray class"""
        cls = super().__new__(mcs, name, bases, class_dict)
        names = [name for name in dir(xr.DataArray)
                 if name not in mcs.__ignore__ and name not in class_dict]
        for name in names:
            setattr(cls, name, property(cls.wrap(name)))
        return cls


class InterpolatedDataArray(metaclass=DataArrayWrapper):
    """Composition of the DataArray and the InterpolatedCoordinates classes 
    that redirects method calls to the DataArray object by default"""

    def __init__(self, data_array, icoords):
        self.data_array = data_array
        self.icoords = icoords

    def __getitem__(self, item):
        data_array = self.data_array[item]
        if not isinstance(item, tuple):
            item = (item,)
        dct = {self.dims[k]: self.icoords[self.dims[k]][item[k]]  # TODO
               for k in range(len(item))}
        icoords = InterpolatedCoordinates(dct)
        return InterpolatedDataArray(data_array, icoords)

    def sel(self, item):
        pass  # TODO

    def isel(self, item):
        pass  # TODO

    def __repr__(self):
        return repr(self.data_array) + "\n" + repr(self.icoords)

    def __str__(self):
        return repr(self)

    def _repr_html_(self):
        html = self.data_array._repr_html_()
        return complete_html(html, self.icoords)

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

    def load_icoords(self):
        """Interpolate coordinates and load them into memory"""
        for dim in self.dims:
            self.data_array.coords[dim] = (
                self.icoords[dim].values())
        return self.data_array


class InterpolatedCoordinates(dict):
    """Subclass of dict that contains interpolated coordinates for each 
    dimension"""

    @property
    def dims(self):
        return self.keys()

    def __repr__(self):
        s = "Interpolated Coordinates:\n"
        for dim, icoord in self.items():
            s += f"  * {dim}".ljust(12)
            s += f"({dim}) "
            s += repr(icoord) + "\n"
        return s
