import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle', radius=0.5):
    """
    Create a radar chart with `num_vars` axes using a custom projection.

    Parameters
    ----------
    num_vars : int
        Number of variables for the radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding the axes.
    radius : float
        Radius of the radar frame in axes coordinates (default is 0.5 to show full 1.0 grid).
    """
    # evenly spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')  # rotate so first axis is at the top

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), radius)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=radius, edgecolor="none")
            else:
                raise ValueError(f"Unknown frame: {frame}")

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(self, 'circle', Path.unit_regular_polygon(num_vars))
                spine.set_transform(
                    Affine2D().scale(radius).translate(0.5, 0.5) + self.transAxes
                )
                return {'polar': spine}
            else:
                raise ValueError(f"Unknown frame: {frame}")

    register_projection(RadarAxes)
    return theta
