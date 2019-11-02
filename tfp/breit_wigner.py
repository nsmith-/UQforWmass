"""The Breit-Wigner distribution class."""

# Dependency imports
import tensorflow as tf
import numpy as np

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import reparameterization as repar
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static

import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = [
    'BreitWigner',
]


class BreitWigner(distribution.Distribution):
    """The Breit-Wigner distribution with mass `mass` and width `width`.

    #### Mathematical details

    The probability density function (pdf) is,

    ```none
    pdf(x; mass, width, smin, smax) = mw/(((x-mass**2)**2+mw**2) * (ymax+ymin))
    mw = mass * width
    ymax = ArcTan((smax - mass**2)/mw)
    ymin = ArcTan((smin - mass**2)/mw)
    ```
    where `mass` is the mass, `width` is the width, `smin` is the minimum
    invariant mass squared, and `smax` is the maximum invariant mass squared.

    #### Examples

    Examples of initialization of one or a batch of distributions.

    ```python
    tfd = tfp.distributions

    # Define a single scalar Breit-Wigner distribution.
    dist = BreitWigner(mass=91.18, width=2.54)

    # Evaluate the cdf at 1, returning a scalar.
    dist.cdf(1.)

    # Define a batch of two scalar valued Breit-Wigner distributions.
    dist = BreitWigner(mass=[91.18,80.358], width=[2.54,2.085])

    # Evaluate the pdf of the first distribution on 90, and the second on 85,
    # returning a length two tensor
    dist.prob([90., 85.])

    # Get 3 samples, returning a 3 x 2 tensor.
    dist.sample([3])

    # Arguments are broadcast when possible.
    # Define a batch of two scalar valued Breit-Wigner distributions.
    # Both have a mass of 91.18, but different widths.
    dist = BreitWigner(mass=91.18, width=[2.54,2.085])

    # Evaluate the pdf of both distributions on the same point, 3.0,
    # returning a length 2 tensor.
    dist.prob(3.)
    ```

    """

    def __init__(self, mass, width, smin=0., smax=13000.**2,
                 validate_args=False, allow_nan_stats=True,
                 name='BreitWigner'):
        """ Construct Breit-Wigner distributions.

        The parameters `mass`, `width`, `smin`, and `smax` must be shaped in a
        way that supports broadcasting (e.g. `mass + width + smin + smax` is a
        valid operation).

        Args:
            mass: Floating point tensor; the mass of the resonance.
                Must contain only positive values.
            width: Floating point tensor; the width of the resonance.
                Must contain only positive values.
            smin: Floating point tensor; the minimum invariant mass squared.
                Must contain only non-negative values.
            smax: Floating point tensor; the maximum invariant mass squared.
                Must contain numbers greater than smin.
            validate_args: Python `bool`, default `False`. When `True`
                distribution parameters are checked for validity despite
                possibly degrading runtime performance. When `False` invalid
                inputs may silently render incorrect outputs.
            allow_nan_stats: Python `bool`, default `True`. When `True`,
                statistics (e.g., mean, mode, variance) use the value '`NaN`'
                to indicate the result is undefined. When `False`, an exception
                is raised if one or more of the statistic's batch members are
                undefined.
            name: Python `str` name prefixed to Ops created by this class.

        Raises:
            TypeError: if `mass`, `width`, `smin`, and `smax` have different
                `dtype`.
        """
        parameters = dict(locals())
        with tf.name_scope(name) as _name:
            dtype = dtype_util.common_dtype(
                [mass, width, smin, smax], tf.float32)
            self._mass = tensor_util.convert_nonref_to_tensor(
                mass, name='mass', dtype=dtype)
            self._width = tensor_util.convert_nonref_to_tensor(
                width, name='width', dtype=dtype)
            self._smin = tensor_util.convert_nonref_to_tensor(
                smin, name='smin', dtype=dtype)
            self._smax = tensor_util.convert_nonref_to_tensor(
                smax, name='smax', dtype=dtype)
            super().__init__(
                dtype=self._mass.dtype,
                reparameterization_type=repar.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=_name,
            )

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(
            zip(('mass', 'width', 'smin', 'smax'),
                ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 4)))

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    @classmethod
    def _params_event_ndims(cls):
        return dict(mass=0, width=0)

    @property
    def mass(self):
        """Distribution parameter for the mass."""
        return self._mass

    @property
    def width(self):
        """Distribution parameter for the width."""
        return self._width

    @property
    def mw(self):
        """Distribution parameter for the mass * width."""
        return self.mass * self.width

    @property
    def smin(self):
        """Distribution parameter for the minimum invariant mass squared."""
        return self._smin

    @property
    def smax(self):
        """Distribution parameter for the maximum invariant mass squared."""
        return self._smax

    @property
    def ymin(self):
        """Distribution parameter for the minimum y value."""
        return tf.atan((self.smin - self.mass**2)/self.mw)

    @property
    def ymax(self):
        """Distribution parameter for the maximum y value."""
        return tf.atan((self.smax - self.mass**2)/self.mw)

    def _batch_shape_tensor(self, mass=None, width=None):
        return prefer_static.broadcast_shape(
            prefer_static.shape(self.mass if mass is None else mass),
            prefer_static.shape(self.width if width is None else width))

    def _batch_shape(self):
        return tf.broadcast_static_shape(self.mass.shape, self.width.shape)

    def _prob(self, x):
        x = tf.convert_to_tensor(x)
        mass = tf.convert_to_tensor(self.mass)
        mass2 = mass**2
        mw = tf.convert_to_tensor(self.mw)
        ymax = tf.convert_to_tensor(self.ymax)
        ymin = tf.convert_to_tensor(self.ymin)
        return mw/(((x-mass2)**2+mw**2) * (ymax-ymin))

    def _log_prob(self, x):
        return tf.math.log(self.prob(x))

    def _cdf(self, x):
        mass = tf.convert_to_tensor(self.mass)
        mass2 = mass**2
        mw = tf.convert_to_tensor(self.mw)
        ymax = tf.convert_to_tensor(self.ymax)
        ymin = tf.convert_to_tensor(self.ymin)
        return ((tf.atan((x-mass2)/self.mw) - self.ymin)
                / (self.ymax - self.ymin))

    def _sample_n(self, n, seed=None):
        mass = tf.convert_to_tensor(self.mass)
        mass2 = mass**2
        mw = tf.convert_to_tensor(self.mw)
        ymin = tf.convert_to_tensor(self.ymin)
        ymax = tf.convert_to_tensor(self.ymax)

        batch_shape = self._batch_shape_tensor(mass=mass, width=mw)
        shape = tf.concat([[n], batch_shape], 0)
        probs = tf.random.uniform(
            shape=shape, minval=0., maxval=1., dtype=self.dtype, seed=seed)
        return mass2 + mw * tf.tan(ymin + probs * (ymax - ymin))

    def _mode(self):
        return self.mass * tf.ones_like(self.mass)

    def _mean(self):
        if self.allow_nan_stats:
            return tf.fill(self.batch_shape_tensor(),
                           dtype_util.as_numpy_dtype(self.dtype)(np.nan))
        else:
            raise ValueError(
                '`mean` is undefined for Breit-Wigner distribution.')

    def _stddev(self):
        if self.allow_nan_stats:
            return tf.fill(self.batch_shape_tensor(),
                           dtype_util.as_numpy_dtype(self.dtype)(np.nan))
        else:
            raise ValueError(
                '`stddev` is undefined for Breit-Wigner distribution.')

    def _parameter_control_dependencies(self, is_init):
        if not self.validate_args:
            return []
        assertions = []
        if is_init != tensor_util.is_ref(self.mass):
            assertions.append(assert_util.assert_positive(
                self.mass, message='Argument `mass` must be positive.'))
        if is_init != tensor_util.is_ref(self.width):
            assertions.append(assert_util.assert_positive(
                self.width, message='Argument `width` must be positive.'))
        if is_init != tensor_util.is_ref(self.smin):
            assertions.append(assert_util.assert_non_negative(
                self.smin,
                message='Argument `smin` must be positive or zero.'))
        if is_init != tensor_util.is_ref(self.smax):
            assertions.append(assert_util.assert_greater(
                self.smax, self.smin,
                message='Argument `smax` must be larger than `smin`.'))
        return assertions


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    np_masses = np.array([81], dtype=np.float64)
    np_widths = np.array([2], dtype=np.float64)
    bw = BreitWigner(np_masses, np_widths)
    mass2 = bw.sample(10000)

    masses = tf.sqrt(mass2)
    plt.hist(masses[:, 0], bins=np.linspace(30, 300, 100))
#    plt.hist(masses[:, 1], bins=np.linspace(30, 300, 100))
#    plt.hist(masses[:, 2], bins=np.linspace(30, 300, 100))
    plt.yscale('log')
    plt.show()

    x = np.array([np.linspace(0, 10000, 10000)]).T
    probs = bw.prob(x**2)

    print(tf.reduce_sum(probs*2*x))

    def BW(mass, width):
        return tfd.TransformedDistribution(distribution=tfd.Cauchy(loc=mass*mass,
                                                                   scale=mass*width),
                                           bijector=tfb.Invert(tfb.Square()))

    bw = BW(81, 2)
    probs = bw.prob(x)
    print(tf.reduce_sum(probs))

    plt.plot(x, probs)
#    plt.yscale('log')
    plt.show()

    cdfs = bw.cdf(x**2)

    plt.plot(x, cdfs)
    plt.show()
