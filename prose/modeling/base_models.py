import numpy as np

class Model(object):
    """
    An abstract class implementing the skeleton of the modeling protocol
    Initial parameter values can either be provided as arguments in the same
    order as ``parameter_names`` or by name as keyword arguments.
    A minimal subclass of this would have the form:
    .. code-block:: python
        class CustomModel(Model):
            parameter_names = ("parameter_1", "parameter_2")
            def get_value(self, x):
                return self.parameter_1 + self.parameter_2 + x
        model = CustomModel(parameter_1=1.0, parameter_2=2.0)
        # or...
        model = CustomModel(1.0, 2.0)
    Args:
        bounds (Optional[list or dict]): Bounds can be given for each
            parameter setting their minimum and maximum allowed values.
            This parameter can either be a ``list`` (with length
            ``full_size``) or a ``dict`` with named parameters. Any parameters
            that are omitted from the ``dict`` will be assumed to have no
            bounds. These bounds can be retrieved later using the
            :func:`celerite.Model.get_parameter_bounds` method and, by
            default, they are used in the :func:`celerite.Model.log_prior`
            method.
    """

    parameter_names = tuple()

    def __init__(self, *args, **kwargs):
        self.unfrozen_mask = np.ones(self.full_size, dtype=bool)
        self.dirty = True

        # Deal with bounds
        self.parameter_bounds = []
        bounds = kwargs.pop("bounds", dict())
        try:
            # Try to treat 'bounds' as a dictionary
            for name in self.parameter_names:
                self.parameter_bounds.append(bounds.get(name, (None, None)))
        except AttributeError:
            # 'bounds' isn't a dictionary - it had better be a list
            self.parameter_bounds = list(bounds)
        if len(self.parameter_bounds) != self.full_size:
            raise ValueError("the number of bounds must equal the number of "
                             "parameters")
        if any(len(b) != 2 for b in self.parameter_bounds):
            raise ValueError("the bounds for each parameter must have the "
                             "format: '(min, max)'")

        # Parameter values can be specified as arguments or keywords
        if len(args):
            if len(args) != self.full_size:
                raise ValueError("expected {0} arguments but got {1}"
                                 .format(self.full_size, len(args)))
            if len(kwargs):
                raise ValueError("parameters must be fully specified by "
                                 "arguments or keyword arguments, not both")
            self.parameter_vector = args

        else:
            # Loop over the kwargs and set the parameter values
            params = []
            for k in self.parameter_names:
                v = kwargs.pop(k, None)
                if v is None:
                    raise ValueError("missing parameter '{0}'".format(k))
                params.append(v)
            self.parameter_vector = params

            if len(kwargs):
                raise ValueError("unrecognized parameter(s) '{0}'"
                                 .format(list(kwargs.keys())))

        # Check the initial prior value
        quiet = kwargs.get("quiet", False)
        if not quiet and not np.isfinite(self.log_prior()):
            raise ValueError("non-finite log prior value")

    def get_value(self, *args, **kwargs):
        """
        Compute the "value" of the model for the current parameters
        This method should be overloaded by subclasses to implement the actual
        functionality of the model.
        """
        raise NotImplementedError("overloaded by subclasses")

    def compute_gradient(self, *args, **kwargs):
        """
        Compute the "gradient" of the model for the current parameters
        The default implementation computes the gradients numerically using
        a first order forward scheme. For better performance, this method
        should be overloaded by subclasses. The output of this function
        should be an array where the first dimension is ``full_size``.
        """
        _EPS = 1.254e-5
        vector = self.get_parameter_vector()
        value0 = self.get_value(*args, **kwargs)
        grad = np.empty([len(vector)] + list(value0.shape), dtype=np.float64)
        for i, v in enumerate(vector):
            vector[i] = v + _EPS
            self.set_vector(vector)
            value = self.get_value(*args, **kwargs)
            vector[i] = v
            self.set_vector(vector)
            grad[i] = (value - value0) / _EPS
        return grad

    def get_gradient(self, *args, **kwargs):
        include_frozen = kwargs.pop("include_frozen", False)
        g = self.compute_gradient(*args, **kwargs)
        if include_frozen:
            return g
        return g[self.unfrozen_mask]

    def __len__(self):
        return self.vector_size

    def _get_name(self, name_or_index):
        try:
            int(name_or_index)
        except (TypeError, ValueError):
            return name_or_index
        return self.get_parameter_names()[int(name_or_index)]

    def __getitem__(self, name_or_index):
        return self.get_parameter(self._get_name(name_or_index))

    def __setitem__(self, name_or_index, value):
        return self.set_parameter(self._get_name(name_or_index), value)

    @property
    def full_size(self):
        """The total number of parameters (including frozen parameters)"""
        return len(self.parameter_names)

    @property
    def vector_size(self):
        """The number of active (or unfrozen) parameters"""
        return self.unfrozen_mask.sum()

    @property
    def parameter_vector(self):
        """An array of all parameters (including frozen parameters)"""
        return np.array([getattr(self, k) for k in self.parameter_names])

    @parameter_vector.setter
    def parameter_vector(self, v):
        if len(v) != self.full_size:
            raise ValueError("dimension mismatch")
        for k, val in zip(self.parameter_names, v):
            setattr(self, k, float(val))
        self.dirty = True

    def get_parameter_dict(self, include_frozen=False):
        """
        Get an ordered dictionary of the parameters
        Args:
            include_frozen (Optional[bool]): Should the frozen parameters be
                included in the returned value? (default: ``False``)
        """
        return OrderedDict(zip(
            self.get_parameter_names(include_frozen=include_frozen),
            self.get_parameter_vector(include_frozen=include_frozen),
        ))

    def get_parameter_names(self, include_frozen=False):
        """
        Get a list of the parameter names
        Args:
            include_frozen (Optional[bool]): Should the frozen parameters be
                included in the returned value? (default: ``False``)
        """
        if include_frozen:
            return self.parameter_names
        return tuple(p
                     for p, f in zip(self.parameter_names, self.unfrozen_mask)
                     if f)

    def get_parameter_bounds(self, include_frozen=False):
        """
        Get a list of the parameter bounds
        Args:
            include_frozen (Optional[bool]): Should the frozen parameters be
                included in the returned value? (default: ``False``)
        """
        if include_frozen:
            return self.parameter_bounds
        return list(p
                    for p, f in zip(self.parameter_bounds, self.unfrozen_mask)
                    if f)

    def get_parameter_vector(self, include_frozen=False):
        """
        Get an array of the parameter values in the correct order
        Args:
            include_frozen (Optional[bool]): Should the frozen parameters be
                included in the returned value? (default: ``False``)
        """
        if include_frozen:
            return self.parameter_vector
        return self.parameter_vector[self.unfrozen_mask]

    def set_parameter_vector(self, vector, include_frozen=False):
        """
        Set the parameter values to the given vector
        Args:
            vector (array[vector_size] or array[full_size]): The target
                parameter vector. This must be in the same order as
                ``parameter_names`` and it should only include frozen
                parameters if ``include_frozen`` is ``True``.
            include_frozen (Optional[bool]): Should the frozen parameters be
                included in the returned value? (default: ``False``)
        """
        v = self.parameter_vector
        if include_frozen:
            v[:] = vector
        else:
            v[self.unfrozen_mask] = vector
        self.parameter_vector = v
        self.dirty = True

    def check_parameter_vector(self, vector):
        # Save the original parameter vector and dirtiness
        vector0 = np.array(self.get_parameter_vector())
        dirty0 = self.dirty

        # Update the vector and compute the prior
        self.set_parameter_vector(vector)
        lp = self.log_prior()

        # Reset the parameter vector
        self.set_parameter_vector(vector0)
        self.dirty = dirty0
        return np.isfinite(lp)

    def freeze_parameter(self, name):
        """
        Freeze a parameter by name
        Args:
            name: The name of the parameter
        """
        i = self.get_parameter_names(include_frozen=True).index(name)
        self.unfrozen_mask[i] = False

    def thaw_parameter(self, name):
        """
        Thaw a parameter by name
        Args:
            name: The name of the parameter
        """
        i = self.get_parameter_names(include_frozen=True).index(name)
        self.unfrozen_mask[i] = True

    def freeze_all_parameters(self):
        """Freeze all parameters of the model"""
        self.unfrozen_mask[:] = False

    def thaw_all_parameters(self):
        """Thaw all parameters of the model"""
        self.unfrozen_mask[:] = True

    def get_parameter(self, name):
        """
        Get a parameter value by name
        Args:
            name: The name of the parameter
        """
        i = self.get_parameter_names(include_frozen=True).index(name)
        return self.get_parameter_vector(include_frozen=True)[i]

    def set_parameter(self, name, value):
        """
        Set a parameter value by name
        Args:
            name: The name of the parameter
            value (float): The new value for the parameter
        """
        i = self.get_parameter_names(include_frozen=True).index(name)
        v = self.get_parameter_vector(include_frozen=True)
        v[i] = value
        self.set_parameter_vector(v, include_frozen=True)

    def log_prior(self):
        """Compute the log prior probability of the current parameters"""
        for p, b in zip(self.parameter_vector, self.parameter_bounds):
            if b[0] is not None and p < b[0]:
                return -np.inf
            if b[1] is not None and p > b[1]:
                return -np.inf
        return 0.0

    @staticmethod
    def parameter_sort(f):
        def func(self, *args, **kwargs):
            values = f(self, *args, **kwargs)
            names = self.get_parameter_names(include_frozen=True)
            ret = [values[k] for k in names]
            # Horrible hack to only return numpy array if that's what was
            # given by the wrapped function.
            if len(ret) and type(ret[0]).__module__ == np.__name__:
                return np.vstack(ret)
            return ret
        return func