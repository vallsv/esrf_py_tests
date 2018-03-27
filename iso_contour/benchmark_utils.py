import timeit
import collections
import numpy
import collections
import os
import numpy
from matplotlib import pyplot
import pylab
import skimage.transform

ROOT = os.getcwd()


class Problem(object):

    def __init__(self, image, mask=None, values=None, complexity=None):
        self._image = image
        self._mask = mask
        if values is None:
            values = image.min() + numpy.array(range(10)) * (image.max() - image.min())
            values = values / len(values)
            values = values[1:-1]
        self._values = values
        if complexity is None:
            complexity = image.shape[0] * image.shape[1]
        self._complexity = complexity

    @property
    def image(self):
        return self._image

    @property
    def mask(self):
        return self._mask

    @property
    def values(self):
        return self._values

    @property
    def complexity(self):
        return self._complexity

    def cropped(self, size, location):
        original = self.image.shape
        if location == "up-right":
            xx = slice(0, size[0])
            yy = slice(original[1] - size[1], original[1])
        else:
            assert("Location not supported")

        image = self.image[xx, yy]
        mask = self.mask[xx, yy]
        coef = (size[0] / self.image.shape[0] + size[1] / self.image.shape[1]) * 0.5
        return Problem(image, mask, self.values, self.complexity * coef)

    def scaled(self, shape):
        coef = (shape[0] / self.image.shape[0] + shape[1] / self.image.shape[1]) * 0.5
        result = Problem(
            image=skimage.transform.resize(self.image, shape),
            mask=None,
            values=self.values,
            complexity=self.complexity * coef)
        return result


def tiled_problem(problem, shape, count):
    """Create a zero array of the saze `shape`, containing `count` times
    the pattern `problem.image`."""
    tiled = numpy.zeros(shape, dtype=numpy.float32)
    pattern = problem.image
    available = numpy.array(tiled.shape) // numpy.array(pattern.shape)
    if count > available[0] * available[1]:
        raise RuntimeError("Not enougth space")
    i = 0
    for y in range(0, tiled.shape[0], pattern.shape[0]):
        if i >= count:
            break
        for x in range(0, tiled.shape[1], pattern.shape[1]):
            if i >= count:
                break
            tiled[y:y + pattern.shape[0], x:x + pattern.shape[1]] = pattern
            i += 1

    result = Problem(image=tiled,
                     mask=None,
                     values=problem.values,
                     complexity=count)
    return result


def create_test_problem():
    import fabio
    data = fabio.open(ROOT + "/data/data.tif").data
    mask = fabio.open(ROOT + "/data/mask.tif").data
    mask = mask == 0
    values = range(10, 1000, int(240/6))[0:7]
    return Problem(data, mask, values, None)

def create_wos_problem():
    # Problem containing a WOS XPad with pixel displacment and mask
    import fabio
    data = numpy.load(ROOT + "/data/wos_tth.npz")
    image = data["tth"]
    mask = fabio.open(ROOT + "/data/wos_mask.edf").data
    mask = (mask != 0)
    values = data["angles"]
    return Problem(image, mask, values, None)

def create_id22_17_problem():
    # Problem containing an image of 4096x4096 with 17 rings
    data = numpy.load(ROOT + "/data/id22_17.npz")
    image = data["tth"]
    mask = (data["mask"] != 0)
    values = data["angles"]
    return Problem(image, mask, values, None)

def create_id22_1441_problem():
    # Problem containing an image of 4096x4096 with 1441 rings
    data = numpy.load(ROOT + "/data/id22_1441.npz")
    image = data["tth"]
    mask = (data["mask"] != 0)
    values = data["angles"]
    return Problem(image, mask, values, None)

def show_problems(problems):
    from matplotlib import pyplot
    pyplot.figure()
    for i, p in enumerate(problems):
        pyplot.subplot(1, len(problems), i + 1)

        pyplot.imshow(p.image)

        if p.mask is not None:
            mask = numpy.ma.masked_where(p.mask == 0, p.mask)
            pyplot.imshow(mask, cmap="cool", alpha=.5)
    pyplot.show()


TimeCollect = collections.namedtuple('TimeCollect', ['image_size',
                                                     'nb_polygons',
                                                     'nb_pixels',
                                                     'nb_points',
                                                     'problem_complexity',
                                                     'algorithm_name',
                                                     'precache_t', 'precache_dt',
                                                     'postprocess_t', 'postprocess_dt',
                                                     'compute_t', 'compute_dt'])

def get_t_dt(array):
    array = numpy.array(array)
    return array.mean(), array.max() - array.mean()


def collect_computation(collected_result, algorithm_name, algorithm_factory, problem):
    values = problem.values

    algo = algorithm_factory(problem.image, problem.mask)
    result = []
    scope = dict(globals())
    scope.update(locals())

    compute_time = timeit.repeat("[result.append(algo.iso_contour(v)) for v in values]", number=10, globals=scope)
    compute_time = get_t_dt(compute_time)
    nb_pixels = problem.image.shape[0] * problem.image.shape[1]
    nb_points, nb_polygons = 0, 0
    for r in result:
        nb_polygons += len(r)
        for p in r:
            nb_points += len(p)
    t = TimeCollect(algorithm_name=algorithm_name,
                    image_size=problem.image.shape,
                    problem_complexity=problem.complexity,
                    nb_pixels=nb_pixels,
                    nb_polygons=nb_polygons,
                    nb_points=nb_points,
                    precache_t=0,
                    precache_dt=0,
                    postprocess_t=0,
                    postprocess_dt=0,
                    compute_t=compute_time[0],
                    compute_dt=compute_time[1])
    collected_result.append(t)


def get_algorithms(collected_result):
    result = set([r.algorithm_name for r in collected_result])
    result = sorted(result)
    return result


def get_result_per_algorithm(collected_result, algorithm_name):
    result = [r for r in collected_result if r.algorithm_name == algorithm_name]
    result = sorted(result, key=lambda r: r.image_size)
    return result


def plot_computation(collected_result, styles=None, by_complexity=False):
    algorithms = get_algorithms(collected_result)

    if by_complexity:
        xlegend = "x-axis: problem complexity"
    else:
        xlegend = "x-axis: number of pixels"

    style = {}
    pyplot.figure()
    for i, algorithm in enumerate(algorithms):
        result = get_result_per_algorithm(collected_result, algorithm)
        if by_complexity:
            x = numpy.array([r.problem_complexity for r in result]) + 1
        else:
            x = numpy.array([r.nb_pixels for r in result]) + 1
        compute_y = numpy.array([r.compute_t for r in result])
        compute_error = numpy.array([r.compute_dt for r in result])
        if styles is not None:
            style = styles.get(algorithm, {})
        if "label" not in style:
            style["label"] = "%s" % (algorithm)
        pyplot.errorbar(x=x, y=compute_y, yerr=compute_error, **style)

    pylab.legend()
    # pyplot.gca().set_xscale("log", nonposx='clip')
    pylab.xlabel("Time spent (%s)" % xlegend)
    pyplot.legend
    pyplot.show()


def plot_computation_per_pixels(collected_result, styles=None, by_complexity=False):
    algorithms = get_algorithms(collected_result)

    if by_complexity:
        xlegend = "x-axis: problem complexity"
    else:
        xlegend = "x-axis: number of pixels"

    style = {}
    pyplot.figure()
    for i, algorithm in enumerate(algorithms):
        result = get_result_per_algorithm(collected_result, algorithm)
        if by_complexity:
            x = numpy.array([r.problem_complexity for r in result]) + 1
        else:
            x = numpy.array([r.nb_pixels for r in result]) + 1
        nb_pixels = numpy.array([r.nb_pixels for r in result])
        compute_y = numpy.array([r.compute_t for r in result]) / nb_pixels * 1000000
        compute_error = numpy.array([r.compute_dt for r in result]) / nb_pixels * 1000000
        if styles is not None:
            style = styles.get(algorithm, {})
        if "label" not in style:
            style["label"] = "%s" % (algorithm)
        pyplot.errorbar(x=x, y=compute_y, yerr=compute_error, **style)

    pylab.legend()
    # pyplot.gca().set_xscale("log", nonposx='clip')
    pylab.xlabel("Time spent per pixel (micro-second) (%s)" % xlegend)
    pyplot.legend
    pyplot.show()


def plot_computation_per_points(collected_result, styles=None, by_complexity=False):
    algorithms = get_algorithms(collected_result)

    if by_complexity:
        xlegend = "x-axis: problem complexity"
    else:
        xlegend = "x-axis: number of pixels"

    style = {}
    pyplot.figure()
    for i, algorithm in enumerate(algorithms):
        result = get_result_per_algorithm(collected_result, algorithm)
        if by_complexity:
            x = numpy.array([r.problem_complexity for r in result]) + 1
        else:
            x = numpy.array([r.nb_pixels for r in result]) + 1
        nb_points = numpy.array([r.nb_points for r in result])
        compute_y = numpy.array([r.compute_t for r in result]) / nb_points * 1000000
        compute_error = numpy.array([r.compute_dt for r in result]) / nb_points * 1000000
        if styles is not None:
            style = styles.get(algorithm, {})
        if "label" not in style:
            style["label"] = "%s" % (algorithm)
        pyplot.errorbar(x=x, y=compute_y, yerr=compute_error, **style)

    pylab.legend()
    # pyplot.gca().set_xscale("log", nonposx='clip')
    pylab.xlabel("Time spent per output points (micro-second)\n( (%s)" % xlegend)
    pyplot.legend
    pyplot.show()

def imshow_problem(problem, marching_square, color_per_polygons=False):
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib import pyplot

    ax = pyplot.gca()
    ax.set_xmargin(0.1)
    ax.set_ymargin(0.1)
    ax.set_ylim([0, problem.image.shape[0]])
    ax.set_xlim([0, problem.image.shape[1]])
    ax.invert_yaxis()

    # image
    pyplot.imshow(problem.image, cmap="Greys", alpha=.5)

    # mask
    if problem.mask is not None:
        mask = numpy.ma.masked_where(problem.mask == 0, problem.mask)
        pyplot.imshow(mask, cmap="cool", alpha=.5)

    # iso contours
    ipolygon = 0
    colors = ["#9400D3", "#4B0082", "#0000FF", "#00FF00", "#FFFF00", "#FF7F00", "#FF0000"]
    for ivalue, value in enumerate(problem.values):
        if not color_per_polygons:
            color = colors[ivalue % len(colors)]
        polygons = marching_square.iso_contour(value)
        for p in polygons:
            if color_per_polygons:
                color = colors[ipolygon % len(colors)]
            ipolygon += 1
            if len(p) == 0:
                continue
            is_closed = numpy.allclose(p[0], p[-1])
            p = Polygon(p, fill=False, edgecolor=color, closed=is_closed)
            ax.add_patch(p)

def plot_problem(problem, marching_square):
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib import pyplot

    pyplot.figure()
    imshow_problem(problem, marching_square)
    pyplot.show()
