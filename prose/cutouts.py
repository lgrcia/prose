from .utils import z_scale
from .utils import index_binning, binning
from .core.image import Image
from .core.source import PointSource
import matplotlib.pyplot as plt
import numpy as np

class Cutout:
    def __init__(self, image, coords=None, shape=(50, 50), fill=np.nan):
        self.sources = []

        # from Image
        if isinstance(image, Image):
            assert coords is not None, "cutout coordinates must be specified if a prose.Image is provided"
            assert shape is not None, "cutout shape must be specified if a prose.Image is provided"
            # get image.data cutout
            self.data = np.zeros(shape)
            p = np.max(shape)
            pad_data = np.pad(image.data, p, constant_values=(fill, fill))
            dx, dy = (np.array(coords[::-1] + p) + (np.array(shape)[:, None]/2 * [-1, 1]).T).T.astype(int)
            self.data = pad_data[dx[0]:dx[1], dy[0]:dy[1]]
            self.shape = np.array(shape)
            self.origin = np.array([dy[0], dx[0]]) - p
        
            # get sources
            sources_in = np.all(np.abs(image.stars_coords - coords) < self.shape[::-1]/2, 1)
            sources = image._sources[sources_in]
            
            for s in sources:
                _s = s.copy()
                _s.coords = _s.coords - self.origin
                self.sources.append(_s)
        else:
            self.shape = np.array(image.shape)
            self.data = image
            self.sources.append(PointSource(coords=self.shape/2))
            self.origin = np.array([0, 0])

        
    @property
    def center(self):
        return np.array(self.shape)/2
    
    @property
    def central_source(self):
        distance = np.linalg.norm([s.coords - self.center for s in self.sources], 1)
        return self.sources[np.argmin(distance)]
    
    def show(self, sources=False, contrast=0.1, zscale=False, cmap="Greys_r", ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if zscale is False:
            vmin = np.nanmedian(self.data)
            vmax = vmax = vmin*(1+contrast)/(1-contrast)
            _ = ax.imshow(self.data, cmap=cmap, origin="lower",vmin=vmin,vmax=vmax, **kwargs)
        else:
            _ = ax.imshow(z_scale(self.data, contrast), cmap=cmap, origin="lower", **kwargs)
        
        if sources:
            for s in self.sources:
                s.plot()

    def symetric_profile(self, binn=0.5):
        y, x = self.center
        Y, X = np.indices(self.data.shape)
        radii = (np.sqrt((X - x) ** 2 + (Y - y) ** 2)).flatten()
        d, values = self.profile(radii)
        return binning(d, values, binn, mean_method=np.median)
    
    def profile(self, d):
        idxs = np.argsort(d)
        _d = d[idxs]
        pixels = self.data.flatten()
        pixels = pixels[idxs]
        
        return _d, pixels 
    
    def major_profile(self, binn=1, debug=False):
        p1 = self.center[::-1][:, None, None]
        p2 = (self.central_source.a_extent[1] - self.origin)[:, None, None]
        Y, X = np.indices(self.data.shape)
        p3 = np.array([X, Y])
        
        # projection
        # https://stackoverflow.com/questions/61341712/calculate-projected-point-location-x-y-on-given-line-startx-y-endx-y
        l2 = np.sum((p1-p2)**2)
        assert l2 != 0, 'p1 and p2 are the same points'
        distances = (np.sum((p3 - p1) * (p2 - p1), 0) / np.sqrt(l2))
        flat_distance = distances.flatten()
        idxs = index_binning(flat_distance, binn)
        distance = np.array([flat_distance[i].mean() for i in idxs])
        values =  np.array([np.nanmax(self.data.flatten()[i]) for i in idxs])
        
        if debug:
            D = np.zeros(self.data.flatten().shape)
            for i, j in enumerate(idxs):
                D[j] = i
            plt.figure()
            plt.imshow(np.reshape(D, self.data.shape), origin="lower")

        return distance, values
    
    def plot_axes(self, ax=None, c="w", alpha=0.2, lw=1, **kwargs):
        xlim, ylim = plt.xlim(), plt.ylim()
        plt.axline(* self.central_source.vertexes, c=c, alpha=alpha, lw=lw, **kwargs)
        plt.axline(*self.central_source.co_vertexes, c=c, alpha=alpha, lw=lw, **kwargs)
        plt.xlim(xlim); plt.ylim(ylim)
        
    def mask(self, central=False, single=False):
        mask = np.zeros_like(self.data)
        if single:
            sources = [self.central_source]
        else:
            sources = [s for s in self.sources if s != self.central_source or central]
        for s in sources:
            _mask = s.elliptical_aperture(1).to_mask(method='center')
            mask[_mask.bbox.iymin:_mask.bbox.iymax, _mask.bbox.ixmin:_mask.bbox.ixmax] = _mask.data

        return mask
    
    def to_source(self, lower_percentile=30):
        mask = self.data > np.percentile(self.data, lower_percentile)
        