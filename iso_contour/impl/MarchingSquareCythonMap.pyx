import cython
import numpy
cimport numpy as cnumpy

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference
from libc.math cimport fabs
cdef double EPSILON = numpy.finfo(numpy.float64).eps

cdef cnumpy.int8_t[2] *EDGE_TO_POINT = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
cdef cnumpy.int8_t[5] *CELL_TO_EDGE = [
                                       # array of index containing
                                       # id0: number of segments (up to 2)
                                       # id1: index of the start of the 1st edge
                                       # id2: index of the end of the 1st edge
                                       # id3: index of the start of the 2nd edge
                                       # id4: index of the end of the 2nd edge
                                       [0, 0, 0, 0, 0],  # Case 0: 0000: nothing
                                       [1, 0, 3, 0, 0],  # Case 1: 0001
                                       [1, 0, 1, 0, 0],  # Case 2: 0010
                                       [1, 1, 3, 0, 0],  # Case 3: 0011

                                       [1, 1, 2, 0, 0],  # Case 4: 0100
                                       [2, 0, 1, 2, 3],  # Case 5: 0101 > ambiguous
                                       [1, 0, 2, 0, 0],  # Case 6: 0110
                                       [1, 2, 3, 0, 0],  # Case 7: 0111

                                       [1, 2, 3, 0, 0],  # Case 8: 1000
                                       [1, 0, 2, 0, 0],  # Case 9: 1001
                                       [2, 0, 3, 1, 2],  # Case 10: 1010 > ambiguous
                                       [1, 1, 2, 0, 0],  # Case 11: 1011

                                       [1, 1, 3, 0, 0],  # Case 12: 1100
                                       [1, 0, 1, 0, 0],  # Case 13: 1101
                                       [1, 0, 3, 0, 0],  # Case 14: 1110
                                       [0, 0, 0, 0, 0],  # Case 15: 1111
                                      ]

cdef struct coord_t:
    cnumpy.int32_t x
    cnumpy.int32_t y

cdef union hashable_coord_t:
    coord_t data
    cnumpy.int64_t hash

cdef struct next_segment_t:
    coord_t pos
    int index
    int edge


cdef class MarchingSquareCythonMap(object):

    cdef cnumpy.float32_t[:, :] _image
    cdef cnumpy.int8_t[:, :] _mask
    cdef unordered_map[cnumpy.int64_t, cnumpy.uint8_t] *_indexes
    cdef vector[cnumpy.float32_t] _forward_points
    cdef vector[cnumpy.float32_t] _backward_points

    def __init__(self, image, mask=None):
        cdef:
            int wh
        self._image = numpy.ascontiguousarray(image, numpy.float32)
        if mask is not None:
            self._mask = numpy.ascontiguousarray(mask, numpy.int8)
            assert(image.shape == mask.shape)
        else:
            self._mask = None
        with nogil:
            wh = self._image.shape[0] + self._image.shape[1]
            self._forward_points.reserve(wh)
            self._backward_points.reserve(wh)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _create_marching_squares(self, double isovalue):
        cdef:
            bint do_mask = self._mask is not None
            int dim_y = self._image.shape[0]
            int dim_x = self._image.shape[1]
            hashable_coord_t coord
            int x, y, i_segment, i_side, i_edge, index, indexes_count = 0
            double tmpf
        with nogil:
            self._indexes = new unordered_map[cnumpy.int64_t, cnumpy.uint8_t]()
            for y in range(dim_y - 1):
                for x in range(dim_x - 1):

                    # Calculate index.
                    index = 0
                    if self._image[y, x] > isovalue:
                        index += 1
                    if self._image[y, x + 1] > isovalue:
                        index += 2
                    if self._image[y + 1, x + 1] > isovalue:
                        index += 4
                    if self._image[y + 1, x] > isovalue:
                        index += 8

                    # Resolve ambiguity
                    if index == 5 or index == 10:
                        # Calculate value of cell center (i.e. average of corners)
                        tmpf = 0.25 * (self._image[y, x] +
                                       self._image[y, x + 1] +
                                       self._image[y + 1, x] +
                                       self._image[y + 1, x + 1])
                        # If below isovalue, swap
                        if tmpf <= isovalue:
                            if index == 5:
                                index = 10
                            else:
                                index = 5

                    # Cache mask information
                    if do_mask:
                        if self._mask[y, x] > 0:
                            index += 16
                        if self._mask[y, x + 1] > 0:
                            index += 32
                        if self._mask[y + 1, x + 1] > 0:
                            index += 64
                        if self._mask[y + 1, x] > 0:
                            index += 128

                    if index < 16 and index != 0 and index != 15:
                        coord.data.x = x
                        coord.data.y = y
                        dereference(self._indexes)[coord.hash] = index

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _compute_point(self,
                cnumpy.uint_t x,
                cnumpy.uint_t y,
                cnumpy.uint8_t edge,
                double isovalue,
                cnumpy.float32_t *result):
        cdef:
            int dx1, dy1, dx2, dy2
            double fx, fy, ff, weight1, weight2
        # Use these to look up the relative positions of the pixels to interpolate
        dx1, dy1 = EDGE_TO_POINT[edge][0], EDGE_TO_POINT[edge][1]
        dx2, dy2 = EDGE_TO_POINT[edge + 1][0], EDGE_TO_POINT[edge + 1][1]
        # Define "strength" of each corner of the cube that we need
        weight1 = 1.0 / (EPSILON + fabs(self._image[y + dy1, x + dx1] - isovalue))
        weight2 = 1.0 / (EPSILON + fabs(self._image[y + dy2, x + dx2] - isovalue))
        # Apply a kind of center-of-mass method
        fx, fy, ff = 0.0, 0.0, 0.0
        fx += <double> dx1 * weight1;
        fy += <double> dy1 * weight1;
        ff += weight1
        fx += <double> dx2 * weight2;
        fy += <double> dy2 * weight2;
        ff += weight2
        fx /= ff
        fy /= ff
        result[0] = x + fx
        result[1] = y + fy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _compute_next_segment(self,
                hashable_coord_t coord,
                cnumpy.uint8_t index,
                cnumpy.uint8_t edge,
                next_segment_t *result):
        cdef:
            hashable_coord_t next_coord
            int next_edge, next_index
            cnumpy.int8_t *edges

        index = index & 0x0F
        if  index == 0 or index == 15:
            result.pos.x = -1
            return

        # clean up the cache
        if index == 5:
            if edge == 0 or edge == 1:
                # it's the first segment
                index = 2
                dereference(self._indexes)[coord.hash] = 7
            else:
                # it's the second segment
                index = 8
                dereference(self._indexes)[coord.hash] = 13
        elif index == 10:
            if edge == 0 or edge == 3:
                # it's the first segment
                index = 14
                dereference(self._indexes)[coord.hash] = 4
            else:
                # it's the second segment
                index = 4
                dereference(self._indexes)[coord.hash] = 1
        else:
            dereference(self._indexes).erase(coord.hash)

        # next
        if edge == 0:
            next_coord.data.x = coord.data.x
            next_coord.data.y = coord.data.y - 1
        elif edge == 1:
            next_coord.data.x = coord.data.x + 1
            next_coord.data.y = coord.data.y
        elif edge == 2:
            next_coord.data.x = coord.data.x
            next_coord.data.y = coord.data.y + 1
        elif edge == 3:
            next_coord.data.x = coord.data.x - 1
            next_coord.data.y = coord.data.y
        else:
            assert False, "Unexpected behaviour"
        if (next_coord.data.x >= self._image.shape[1] - 1
                or next_coord.data.y >= self._image.shape[0] - 1
                or next_coord.data.x < 0 or next_coord.data.y < 0):
            # out of the indexes
            result.pos.x = -1
            return

        next_index = dereference(self._indexes)[next_coord.hash]
        next_index = next_index & 0x0F
        if next_index == 0 or next_index == 15:
            # nothing anymore
            result.pos.x = -1
            return

        # top became down, up be came down
        from_edge = edge + 2 if edge < 2 else edge - 2
        edges = CELL_TO_EDGE[next_index]
        if next_index == 5 or next_index == 10:
            # the targeted side is not from_side but the other (from the same segment)
            if edges[1] == from_edge:
                next_edge = edges[2]
            elif edges[2] == from_edge:
                next_edge = edges[1]
            elif edges[3] == from_edge:
                next_edge = edges[4]
            elif edges[4] == from_edge:
                next_edge = edges[3]
        else:
            # the targeted side is not from_side but the other
            next_edge = edges[1] if edges[1] != from_edge else edges[2]

        result.pos.x = next_coord.data.x
        result.pos.y = next_coord.data.y
        result.index = next_index
        result.edge = next_edge
        return


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _extract_polygon(self, double isovalue, hashable_coord_t coord):
        cdef:
            int i
            cnumpy.uint8_t index, edge
            cnumpy.float32_t *point = [0, 0]
            next_segment_t first_pos
            next_segment_t next_segment

        self._forward_points.clear()
        self._backward_points.clear()
        index = dereference(self._indexes)[coord.hash]
        index = index & 0x0F

        edge = CELL_TO_EDGE[index][1 + 0]
        first_pos.pos = coord.data
        first_pos.index = index
        first_pos.edge = edge
        self._compute_point(coord.data.x, coord.data.y, edge, isovalue, point)
        self._forward_points.push_back(point[0])
        self._forward_points.push_back(point[1])

        edge = CELL_TO_EDGE[index][1 + 1]
        self._compute_point(coord.data.x, coord.data.y, edge, isovalue, point)
        self._forward_points.push_back(point[0])
        self._forward_points.push_back(point[1])

        while True:
            self._compute_next_segment(coord, index, edge, &next_segment)
            if next_segment.pos.x < 0:
                break
            coord.data, index, edge = next_segment.pos, next_segment.index, next_segment.edge
            self._compute_point(coord.data.x, coord.data.y, edge, isovalue, point)
            self._forward_points.push_back(point[0])
            self._forward_points.push_back(point[1])

        coord.data, index, edge = first_pos.pos, first_pos.index, first_pos.edge
        while True:
            self._compute_next_segment(coord, index, edge, &next_segment)
            if next_segment.pos.x < 0:
                break
            coord.data, index, edge = next_segment.pos, next_segment.index, next_segment.edge
            self._compute_point(coord.data.x, coord.data.y, edge, isovalue, point)
            self._backward_points.push_back(point[1])
            self._backward_points.push_back(point[0])

        result = numpy.empty(self._forward_points.size() + self._backward_points.size(), dtype=numpy.float32)
        if self._backward_points.size() > 0:
            result[self._backward_points.size() - 1::-1] = self._backward_points
        result[self._backward_points.size():] = self._forward_points
        result = result.reshape(-1, 2)
        return result


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _extract_polygons(self, double isovalue):
        cdef:
            int x, y
            hashable_coord_t coord
            cnumpy.uint8_t index
            unordered_map[cnumpy.int64_t, cnumpy.uint8_t].iterator it
        polygons = []
        with nogil:
            while not dereference(self._indexes).empty():
                it = dereference(self._indexes).begin()
                coord.hash = dereference(it).first
                index = dereference(it).second
                index = index & 0x0F
                if index == 0 or index == 15:
                    dereference(self._indexes).erase(coord.hash)
                    continue
                with gil:
                    polygon = self._extract_polygon(isovalue, coord)
                    polygons.append(polygon)

                if index == 5 or index == 10:
                    index = dereference(self._indexes)[dereference(it).first]
                    index = index & 0x0F
                    if index == 0 or index == 15:
                        continue
                    # There is maybe a second polygon to extract
                    with gil:
                        polygon = self._extract_polygon(isovalue, coord)
                        polygons.append(polygon)
                # not available on my gcc...
                # dereference(indexes).erase(it)
                dereference(self._indexes).erase(coord.hash)

        return polygons

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def iso_contour(self, value=None):
        self._create_marching_squares(value)
        polygons = self._extract_polygons(value)
        del self._indexes
        return polygons
