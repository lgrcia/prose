from matplotlib.patches import Ellipse, Circle
from photutils.aperture import *
from photutils.isophote import EllipseGeometry
from photutils.isophote import Ellipse as IsoEllipse
import numpy as np
import matplotlib.pyplot as plt

color = [0.51, 0.86, 1.]

def distance(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))

def clean_stars_positions(positions, tolerance=50):
    keep = []

    distance_to_others = np.array(
        [[distance(v, w) for w in positions] for v in positions]
    )
    for i, _distances in enumerate(distance_to_others):
        _distances[i] = np.inf
        close_stars = np.flatnonzero(_distances < tolerance)
        if len(close_stars) == 0:
            keep.append(i)

    return np.unique(keep)
    
class Source:
    def __init__(self, region=None, coords=None, peak=0, i=None):
        """Representation of a source in an image

        Parameters
        ----------
        region : skimage.measure.RegionProperties, optional
            An skimage RegionProperties containing the source, by default None. Only one of 'coords' or 'region' must be provided
        coords : array, optional
            Pixel coordinates of the source, by default None. Only one of 'coords' or 'region' must be provided
        peak : float, optional
            maximum pixel value of the source, by default 0
        i : int, optional
            index identifier of the source, by default None
        """
        # assert (region is not None) or (coords is not None), "One of 'coords' or 'region' must be provided"
        # if (region is not None) and (coords is not None):
        #     AssertionError("Only one of 'coords' or 'region' must be provided")
        # if (region is not None) and (peak != 0):
        #     AssertionError("Providing 'peak' has no effect when 'region' is provided")

        if region is not None:
            self.a = region.axis_major_length
            self.b = region.axis_minor_length
            self.orientation = region.orientation
            self.coords = np.array(region.centroid_weighted[::-1])
            self.peak = region.intensity_max
        else:
            self.a = 1.
            self.b = 1.
            self.orientation = 0.
            self.coords = coords
            self.peak = peak

        self.i = i
        self.discarded = False
        self._region = region

    @property
    def vertexes(self):
        """Coordinates of the Ellipse vertexes, endpoints of the major axis

        Returns
        -------
        np.array
            vertexes coordinates
        """
        x0, y0 = self.coords
        theta = self.orientation
        x1 = x0 - np.sin(np.pi+theta) * 0.5 * self.a
        y1 = y0 - np.cos(np.pi+theta) * 0.5 * self.a
        x2 = x0 - np.sin(theta) * 0.5 * self.a
        y2 = y0 - np.cos(theta) * 0.5 * self.a

        return np.array([x1, y1]), np.array([x2, y2])

    @property
    def co_vertexes(self):
        """Coordinates of the Ellipse co-vertexes, endpoints of the minor axis

        Returns
        -------
        np.array
            co-vertexes coordinates
        """
        x0, y0 = self.coords
        theta = self.orientation + np.pi/2
        x1 = x0 - np.sin(np.pi+theta) * 0.5 * self.b
        y1 = y0 - np.cos(np.pi+theta) * 0.5 * self.b
        x2 = x0 - np.sin(theta) * 0.5 * self.b
        y2 = y0 - np.cos(theta) * 0.5 * self.b

        return np.array([x1, y1]), np.array([x2, y2])

    def copy(self):
        """Return a copy of the Source

        Returns
        -------
        Source
            copy
        """
        copy = self.__class__()
        copy.a = self.a
        copy.b = self.b
        copy.peak = self.peak
        copy.orientation = self.orientation
        copy.i = self.i
        copy.coords = self.coords
        return copy

    
    def plot_circle(self, radius, c=color, ax=None, label=True, fontsize=12, **kwargs):
        """Plot a circle centered on source

        Parameters
        ----------
        radius : float
            radii of the circle in pixels
        c : str, optional
            color of the circle, by default color
        ax : Axe, optional
            pyplot axe in which to plot the circle, by default None
        label : bool, optional
            whether to display the Source.i index, by default True
        fontsize : int, optional
            Font size for the source index, by default 12
        """
        if ax is None:
            ax = plt.gca()
        circle = Circle(self.coords, radius, fill=None, ec=c, **kwargs)
        ax.add_artist(circle)
        if label and self.i is not None:
            plt.text(*(np.array(self.coords) - [0, 1.5*radius]), self.i, c=c, ha="center", va="top", fontsize=fontsize)
        
    def plot_ellipse(self, n=10, c=color, ax=None, label=True, fontsize=12, **kwargs):
        """Plot an ellipse centered on source, with semi-major/minor length defined by the source itself 

        Parameters
        ----------
        n : float
            offset added to the major and minor axis (major axis of the plotted ellipse will be `Source.a + n`)
        c : str, optional
            color of the circle, by default color
        ax : Axe, optional
            pyplot axe in which to plot the circle, by default None
        label : bool, optional
            whether to display the Source.i index, by default True
        fontsize : int, optional
            Font size for the source index, by default 12
        """
        if ax is None:
            ax = plt.gca()

        ax = plt.gca()
        e = Ellipse(
            xy=self.coords,
            width=n + self.b,
            height=n + self.a,
            angle=-self.orientation * 180. / np.pi,
            **kwargs
        )
        e.set_facecolor('none')
        e.set_edgecolor(c)
        ax.add_artist(e) 
    
        if label and self.i is not None:
            rad = np.pi/2 + self.orientation
            label_coord = self.coords + [0, -(np.abs(self.a*rad/2) + self.b/2)]
            plt.text(*label_coord, self.i, c=c, ha="center", va="top", fontsize=fontsize)

    def circular_aperture(self, r, scale=True):
        """`photutils.aperture.CircularAperture` centered on the source

        Parameters
        ----------
        r : float
            radius
        scale : bool, optional
            whether to scale r to Source.a, by default True

        Returns
        -------
        photutils.aperture.CircularAperture
        """
        if scale:
            radius = r*self.a
        else:
            radius = r
        return CircularAperture(self.coords, radius)

    def elliptical_aperture(self, r, scale=True):
        """`photutils.aperture.EllipticalAperture` centered on the source

        Parameters
        ----------
        r : float
            semi-major axis of the aperture. Semi minor will be `r*Source.b/Source.a`
        scale : bool, optional
            whether to scale r to Source.a, by default True

        Returns
        -------
        photutils.aperture.CircularAperture
        """
        if scale:
            _a, _b = r*self.a, r*self.b
        else:
            _a, _b = r, r*self.b/self.a
        return EllipticalAperture(self.coords, _a, _b, np.pi/2-self.orientation)

    def rectangular_aperture(self, r, scale=True):
        if scale:
            _a, _b = r*self.a, r*self.b
        else:
            _a, _b = r, r*self.b/self.a
        return RectangularAperture(self.coords, _a, _b, np.pi/2-self.orientation)

    def circular_annulus(self, r0, r1, scale=False):
        if scale:
            _r0 = r0*self.a
            _r1 = r1*self.a
        else:
            _r0 = r0
            _r1 = r1
        return CircularAnnulus(self.coords, _r0, _r1)

    def elliptical_annulus(self, r0, r1, scale=False):
        if scale:
            _a0, _b0 = r0*self.a, r0*self.b
            _a1, _b1 = r1*self.a, r1*self.b
        else:
            _a0, _b0 = r0, r0*self.b/self.a
            _a1, _b1 = r1, r1*self.b/self.a
        return EllipticalAnnulus(self.coords, _a0, _a1, _b1, theta=np.pi/2-self.orientation)

    def rectangular_annulus(self, r0, r1, scale=False):
        if scale:
            _a0, _b0 = r0*self.a, r0*self.b
            _a1, _b1 = r1*self.a, r1*self.b
        else:
            _a0, _b0 = r0, r0*self.b/self.a
            _a1, _b1 = r1, r1*self.b/self.a
        return RectangularAnnulus(self.coords, _a0, _a1, _b1, theta=np.pi/2-self.orientation)

    def fit_isophotes(self, debug=False):
        """Fit a photutils.isophote.Ellipse to the source. Requires the source to be instantiated from a skimage RegionProperties

        Parameters
        ----------
        debug : bool, optional
            whether to plot the result for debugging, by default False

        Returns
        -------
        output of photutils.isophote.Ellipse.fit_image
        """
        data = self._region.image_intensity
        y0, x0 = np.unravel_index(np.argmax(data), data.shape)
        e = self.b / self.a
        geometry = EllipseGeometry(x0, y0, sma=self.a/4, eps=e, pa=np.pi/2 - self.orientation)
        ellipse = IsoEllipse(data - np.median(data), geometry)
        isolist = ellipse.fit_image()
        
        if debug:
            plt.imshow(data)
            smas = np.linspace(3, 20, 15)
            for sma in smas:
                iso = isolist.get_closest(sma)
                x, y, = iso.sampled_coordinates()
                plt.plot(x, y, color='white')
                
        return isolist

    @property
    def _symbol(self):
        return "?"

    @property
    def _desc(self):
        return f"{self._symbol} {self.__class__.__name__}" + f" {self.i}" if self.i is not None else ""

    def _repr_dict(self, n=8):
        return {
            "coords": f"{self.coords[0]:.2f}".rjust(n) + f"{self.coords[1]:.2f}".rjust(n),
            "a, b": f"{self.a:.2f}".rjust(n) + f"{self.b:.2f}".rjust(n),
            "e": f"{self.b/self.a:.2f}".rjust(n),
            "region": str(self._region is not None).rjust(n)
            }

    def __str__(self):
        table = "\n".join([f"  {n}".ljust(8) + f"{v}" for n, v in self._repr_dict().items()])
        return f"{self._desc}\n  {'-'*(len(self._desc)-2)}\n{table}"
    
def auto_source(region, i=None, trace=0.3, extended=0.9, discard=False):
    if region is None:
        return DiscardedSource(region, i=i)
    a = region.axis_major_length
    b = region.axis_minor_length
    if a == 0.:
        if discard:
            return DiscardedSource(region, i=i)
        else:
            return PointSource(region, i=i)
    ratio = b/a
    if ratio <= extended:
        if ratio <= trace:
            return TraceSource(region, i=i)
        else:
            return ExtendedSource(region, i=i)
    else:
        return PointSource(region, i=i)

class DiscardedSource(Source):
    def __init__(self, region, i=None):
        super().__init__(region, i=i)
        self.discarded = True
        
    def plot(self, ms=15, c="C0", ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot(*self.coords, "x", c=c, ms=ms, **kwargs)
    
class PointSource(Source):
    def __init__(self, region=None, coords=None, peak=0, i=None):
        """Point source (star)

        Parameters
        ----------
        region : skimage.measure.RegionProperties, optional
            An skimage RegionProperties containing the source, by default None. Only one of 'coords' or 'region' must be provided
        coords : array, optional
            Pixel coordinates of the source, by default None. Only one of 'coords' or 'region' must be provided
        peak : float, optional
            maximum pixel value of the source, by default 0
        i : int, optional
            index identifier of the source, by default None
        """
        super().__init__(region=region, coords=coords, peak=peak, i=i)

    @property
    def _symbol(self):
        return chr(8226)

    def plot(self, radius=15, **kwargs):
        """Plot circle centered on source

        Parameters
        ----------
        radius : int, optional
            radius, by default 15
        """
        self.plot_circle(radius, **kwargs)

    def aperture(self, r=1, scale=True):
        return self.circular_aperture(r, scale=scale)

    def annulus(self, r0=1.05, r1=1.4, scale=True):
        return self.circular_annulus(r0, r1, scale=scale)

    
class ExtendedSource(Source):
    def __init__(self, region=None, coords=None, peak=0, i=None):
        """Extended source (comet, galaxy or lensed source)

        Parameters
        ----------
        region : skimage.measure.RegionProperties, optional
            An skimage RegionProperties containing the source, by default None. Only one of 'coords' or 'region' must be provided
        coords : array, optional
            Pixel coordinates of the source, by default None. Only one of 'coords' or 'region' must be provided
        peak : float, optional
            maximum pixel value of the source, by default 0
        i : int, optional
            index identifier of the source, by default None
        """
        super().__init__(region=region, coords=coords, peak=peak, i=i)

    @property
    def _symbol(self):
        return chr(11053)

    def plot(self, radius=6, **kwargs):
        """Plot Ellipse on source

        Parameters
        ----------
        radius : int, optional
            extension to minor/major axis, by default 6
        """
        self.plot_ellipse(radius, **kwargs)

    def compute_centroid(self, method="max"):
        """Recompute source centroid (inplace). To use this method, source must be obtained from a region 
        """
        if method == "max":
            y0, x0 = np.unravel_index(np.argmax(self._region.image_intensity), self._region.image.shape)
            dy, dx, _, _ = self._region.bbox
            self.coords = np.array([x0+dx, y0+dy])
        elif method == "isophote":
            isolist = self.fit_isophotes()
            origin = np.array(self._region.bbox)[0:2][::-1]
            self.coords = np.array([isolist[0].x0, isolist[0].y0]) + origin

    def aperture(self, r=1, scale=True):
        return self.elliptical_aperture(r, scale=scale)
    
    def annulus(self, r0=1.05, r1=1.4, scale=True):
        return self.elliptical_annulus(r0, r1, scale=scale)


class TraceSource(Source):
    def __init__(self, region=None, coords=None, peak=0, i=None):
        """Trace source (diffracted spectrum, satellite streak or cosmic ray)

        Parameters
        ----------
        region : skimage.measure.RegionProperties, optional
            An skimage RegionProperties containing the source, by default None. Only one of 'coords' or 'region' must be provided
        coords : array, optional
            Coordinates of the source, by default None. Only one of 'coords' or 'region' must be provided
        peak : float, optional
            maximum pixel value of the source, by default 0
        i : int, optional
            index identifier of the source, by default None
        """
        super().__init__(region=region, coords=coords, peak=peak, i=i)
        self.discarded = True
        
    def plot(self, offset=10, ax=None, c=color, label=True, fontsize=12):
        if ax is None:
            ax = plt.gca()
        x0, y0 = self.coords
        theta = self.orientation
        x1 = x0 - np.sin(np.pi+theta) * 0.5 * self.a
        y1 = y0 - np.cos(np.pi+theta) * 0.5 * self.a
        x2 = x0 - np.sin(theta) * 0.5 * self.a
        y2 = y0 - np.cos(theta) * 0.5 * self.a

        ax.plot((x1, x2), (y1, y2), c=c)

        if label and self.i is not None:
            label_coords = self.coords + [0, -offset]
            plt.text(*label_coords, self.i, c=c, ha="center", va="top", fontsize=fontsize)
    
    def aperture(self, r=1, scale=True):
        return self.rectangular_aperture(r, scale=scale)

    def annulus(self, r0=1.05, r1=1.4, scale=True):
        return self.rectangular_annulus(r0, r1, scale=scale)


    