import numpy
cimport numpy as cnumpy

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.list cimport list as clist
from libc.math cimport fabs
from libcpp cimport bool
cimport libc.stdlib
cimport libc.string

cdef extern from "<algorithm>" namespace "std":
    Iter find[Iter, T](Iter first, Iter last, const T& val)

from cython.parallel import prange
from cython.operator cimport dereference
from cython.operator cimport preincrement
cimport cython


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


ctypedef cnumpy.int64_t hash_index_t


cdef struct point_t:
    cnumpy.float32_t x
    cnumpy.float32_t y


cdef cppclass polygon_description_t:
    hash_index_t begin
    hash_index_t end
    int begin_x
    int end_x
    clist[point_t] points

    polygon_description_t() nogil:
        pass

    void insert_to(hash_index_t previous_index, int new_x, hash_index_t new_index, point_t &new_point) nogil:
        if this.begin == previous_index:
            this.points.push_front(new_point)
            this.begin = new_index
            this.begin_x = new_x
        elif this.end == previous_index:
            this.points.push_back(new_point)
            this.end = new_index
            this.end_x = new_x


cdef struct next_segment_t:
    int x
    int y
    int index
    int edge


cdef cppclass TileContext_t:
    int pos_x
    int pos_y
    int dim_x
    int dim_y

    # vector[polygon_description_t*] x_scan
    polygon_description_t** x_scan
    polygon_description_t* y_prev
    polygon_description_t* y_prev2

    clist[polygon_description_t*] final_polygons

    map[hash_index_t, polygon_description_t*] polygons

    TileContext_t() nogil:
        pass

    __dealloc__() nogil:
        if this.x_scan != NULL:
            libc.stdlib.free(this.x_scan)

    void alloc_cache() nogil:
        cdef:
            int size = this.dim_x
        this.x_scan = <polygon_description_t**>libc.stdlib.malloc(size * sizeof(polygon_description_t*))
        libc.string.memset(this.x_scan, 0, size * sizeof(polygon_description_t*))


cdef class MarchingSquareCythonScanInsertOpenMp(object):
    """Marching square using an insertion algorithm to reconstruct polygons
    on the fly while iterating input data.
    """

    cdef cnumpy.float32_t[:, :] _image
    cdef cnumpy.int8_t[:, :] _mask

    cdef cnumpy.float32_t *_image_ptr
    cdef cnumpy.int8_t *_mask_ptr
    cdef int _dim_x
    cdef int _dim_y
    cdef int _group_size
    cdef int _group_mode

    cdef TileContext_t* _final_context

    def __init__(self, image, mask=None, openmp_group_mode="tile", openmp_group_size=256):
        self._image = numpy.ascontiguousarray(image, numpy.float32)
        self._image_ptr = &self._image[0][0]
        if mask is not None:
            assert(image.shape == mask.shape)
            self._mask = numpy.ascontiguousarray(mask, numpy.int8)
            self._mask_ptr = &self._mask[0][0]
        else:
            self._mask = None
            self._mask_ptr = NULL
        self._group_mode = {"tile": 0, "row": 1, "col": 2}[openmp_group_mode]
        self._group_size = openmp_group_size
        with nogil:
            self._dim_y = self._image.shape[0]
            self._dim_x = self._image.shape[1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _marching_squares(self, cnumpy.float64_t isovalue):
        cdef:
            int x, y, i
            vector[TileContext_t*] contexts
            TileContext_t *context
            int dim_x, dim_y

        if self._group_mode == 0:
            for y in range(0, self._dim_y - 1, self._group_size):
                for x in range(0, self._dim_x - 1, self._group_size):
                    context = self._create_context(x, y, self._group_size, self._group_size)
                    contexts.push_back(context)
        elif self._group_mode == 1:
            # row
            for y in range(0, self._dim_y - 1, self._group_size):
                context = self._create_context(0, y, self._dim_x - 1, self._group_size)
                contexts.push_back(context)
        elif self._group_mode == 2:
            # col
            for x in range(0, self._dim_x - 1, self._group_size):
                context = self._create_context(x, 0, self._group_size, self._dim_y - 1)
                contexts.push_back(context)
        else:
            # FIXME: Good to add check
            pass

        # openmp
        #for i in range(contexts.size()):
        for i in prange(contexts.size(), nogil=True):
            self._marching_squares_mp(contexts[i], isovalue)

        if contexts.size() == 1:
            # shortcut
            self._final_context = contexts[0]
            return

        # merge
        with nogil:
            self._final_context = new TileContext_t()
            # self._final_context.polygons.reserve(self._dim_x * 2 + self._dim_y * 2)
            for i in range(contexts.size()):
                self._merge_context(self._final_context, contexts[i])
                del contexts[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef TileContext_t *_create_context(self, int x, int y, int dim_x, int dim_y) nogil:
        cdef:
            TileContext_t *context
        context = new TileContext_t()
        context.pos_x = x
        context.pos_y = y
        context.dim_x = dim_x
        context.dim_y = dim_y
        if x + context.dim_x > self._dim_x - 1:
            context.dim_x = self._dim_x - 1 - x
        if y + context.dim_y > self._dim_y - 1:
            context.dim_y = self._dim_y - 1 - y
        if context.dim_x <= 0 or context.dim_y <= 0:
            del context
            return NULL
        return context

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _marching_squares_mp(self, TileContext_t *context, cnumpy.float64_t isovalue) nogil:
        cdef:
            int x, y, index
            cnumpy.float64_t tmpf
            cnumpy.float32_t *_image_ptr
            cnumpy.int8_t *_mask_ptr
            vector[TileContext_t*] contexts
            int dim_x, dim_y

        context.alloc_cache()
        context.y_prev = NULL

        _image_ptr = self._image_ptr + (context.pos_y * self._dim_x + context.pos_x)
        if self._mask_ptr != NULL:
            _mask_ptr = self._mask_ptr + (context.pos_y * self._dim_x + context.pos_x)
        else:
            _mask_ptr = NULL

        for y in range(context.pos_y, context.pos_y + context.dim_y):
            for x in range(context.pos_x, context.pos_x + context.dim_x):
                # Calculate index.
                index = 0
                if _image_ptr[0] > isovalue:
                    index += 1
                if _image_ptr[1] > isovalue:
                    index += 2
                if _image_ptr[self._dim_x] > isovalue:
                    index += 8
                if _image_ptr[self._dim_x + 1] > isovalue:
                    index += 4

                # Resolve ambiguity
                if index == 5 or index == 10:
                    # Calculate value of cell center (i.e. average of corners)
                    tmpf = 0.25 * (_image_ptr[0] +
                                   _image_ptr[1] +
                                   _image_ptr[self._dim_x] +
                                   _image_ptr[self._dim_x + 1])
                    # If below isovalue, swap
                    if tmpf <= isovalue:
                        if index == 5:
                            index = 10
                        else:
                            index = 5

                # Cache mask information
                if _mask_ptr != NULL:
                    _mask_ptr += 1
                    if _mask_ptr[0] > 0:
                        index += 16
                    if _mask_ptr[1] > 0:
                        index += 32
                    if _mask_ptr[self._dim_x] > 0:
                        index += 128
                    if _mask_ptr[self._dim_x + 1] > 0:
                        index += 64

                if index < 16 and index != 0 and index != 15:
                    self._insert_pattern(context, x, y, index, isovalue)

                _image_ptr += 1

            # There is a missing pixel at the end of each rows
            _image_ptr += self._dim_x - context.dim_x
            if _mask_ptr != NULL:
                _mask_ptr += self._dim_x - context.dim_x

        self._store_unconnected_polygons(context)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _insert_pattern(self, TileContext_t *context, int x, int y, int pattern, cnumpy.float64_t isovalue) nogil:
        cdef:
            int segment
            polygon_description_t *tmp
            polygon_description_t *old_y_prev

        if pattern == 5:
            # that's a real special case to fix the cache with that pattern
            context.y_prev2 = context.y_prev
            self._insert_segment(context, x, y, 0, 1, isovalue)
            # swap previous and new y_prev
            tmp = context.y_prev2
            context.y_prev2 = context.y_prev
            context.y_prev = tmp
            self._insert_segment(context, x, y, 2, 3, isovalue)
            # retitute computation from the first insert_segment
            context.y_prev = context.y_prev2
            context.y_prev2 = NULL
        else:
            for segment in range(CELL_TO_EDGE[pattern][0]):
                begin_edge = CELL_TO_EDGE[pattern][1 + segment * 2 + 0]
                end_edge = CELL_TO_EDGE[pattern][1 + segment * 2 + 1]
                self._insert_segment(context, x, y, begin_edge, end_edge, isovalue)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef hash_index_t _create_hash_index(self, int x, int y, cnumpy.uint8_t edge) nogil:
        """Create an identifier for a tuple x-y-edge (which is reversible)

        There is no way to create hashable struct in cython. Then it uses
        a standard hashable type.

        For example, the tuple (x=0, y=0, edge=2) is equal to (x=1, y=0, edge=0)
        """
        cdef:
            hash_index_t v = 0
        if edge == 2:
            y += 1
            edge = 0
        elif edge == 3:
            x -= 1
            edge = 1
        # Avoid negative values
        x += 1
        y += 1

        v += edge
        v <<= 20
        v += x
        v <<= 20
        v += y
        return v

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _insert_segment(self, TileContext_t *context,
                              int x, int y,
                              cnumpy.uint8_t begin_edge,
                              cnumpy.uint8_t end_edge,
                              cnumpy.float64_t isovalue) nogil:
        cdef:
            int pos
            int i
            point_t point
            hash_index_t begin, end, index
            polygon_description_t *description
            polygon_description_t *description_left
            polygon_description_t *description_top
            vector[polygon_description_t*].iterator it_desc
            map[hash_index_t, polygon_description_t*].iterator it_begin
            map[hash_index_t, polygon_description_t*].iterator it_end
            int edge_sum
            clist[point_t].iterator it_points

        # 0 is always stored in begin
        # 3 is always stored in end
        description = NULL
        begin = self._create_hash_index(x, y, begin_edge)
        end = self._create_hash_index(x, y, end_edge)

        if begin_edge == 0:
            description_top = context.x_scan[x - context.pos_x]
            if description_top != NULL:
                if description_top.begin != begin and description_top.end != begin:
                    # TODO: The poly on top have to be removed
                    description_top = NULL
        else:
            # TODO: The poly on top have to be removed
            description_top = NULL

        if end_edge == 3:
            description_left = context.y_prev
            if description_left != NULL:
                if description_left.begin != end and description_left.end != end:
                    # TODO: This poly on left have to be removed
                    description_left = NULL
        else:
            # TODO: This poly on left have to be removed
            description_left = NULL

        if description_top != NULL and description_left != NULL:
            # Merge polygons from the right and the top
            if description_top == description_left:
                # The segment closes a polygon
                # FIXME: this intermediate assign is not needed
                point = description_left.points.front()
                description_left.points.push_back(point)
                context.final_polygons.push_back(description_left)
                context.y_prev = NULL
                context.x_scan[x - context.pos_x]= NULL
                description = NULL
            else:
                # FIXME: It have to be optimized
                # FIXME: We can recycle a description instead of creating a new one
                description = new polygon_description_t()

                # Make sure the last element of the list is the one to connect
                if description_top.begin == begin:
                    # O(n)
                    description_top.points.reverse()
                    description.begin = description_top.end
                    description.begin_x = description_top.end_x
                else:
                    description.begin = description_top.begin
                    description.begin_x = description_top.begin_x

                # O(1)
                description.points.splice(description.points.end(), description_top.points)

                # Make sure the first element of the list is the one to connect
                if description_left.end == end:
                    description_left.points.reverse()
                    description.end = description_left.begin
                    description.end_x = description_left.begin_x
                else:
                    description.end = description_left.end
                    description.end_x = description_left.end_x

                description.points.splice(description.points.end(), description_left.points)

                # FIXME: We should erase using the iterator

                # Update cache

                context.y_prev = NULL
                context.x_scan[x - context.pos_x] = NULL
                self._replace_from_cache(context, description_left, description)
                self._replace_from_cache(context, description_top, description)

                it_begin = context.polygons.find(description.begin)
                if it_begin != context.polygons.end():
                    dereference(it_begin).second = description
                it_end = context.polygons.find(description.end)
                if it_end != context.polygons.end():
                    dereference(it_end).second = description

                del description_top
                del description_left
                description = NULL

        elif description_top != NULL:
            # Here begin_edge is always connected to the top
            self._compute_point(x, y, end_edge, isovalue, point)
            description_top.insert_to(begin, x, end, point)
            description = description_top
            context.x_scan[x - context.pos_x] = NULL
        elif description_left != NULL:
            # Here end_edge is always connected to the left
            self._compute_point(x, y, begin_edge, isovalue, point)
            description_left.insert_to(end, x, begin, point)
            description = description_left
            context.y_prev = NULL
        else:
            # It's a new description
            description = new polygon_description_t()
            description.begin = begin
            description.begin_x = x
            description.end = end
            description.end_x = x
            self._compute_point(x, y, begin_edge, isovalue, point)
            description.points.push_back(point)
            self._compute_point(x, y, end_edge, isovalue, point)
            description.points.push_back(point)

        if description == NULL:
            return

        edge_sum = (1 << begin_edge) | (1 << end_edge)
        if (edge_sum & 0x1) != 0 and y == context.pos_y:
            # store
            if begin_edge == 0:
                index = begin
            elif end_edge == 0:
                index = end
            context.polygons[index] = description
        elif (edge_sum & 0x2) != 0 and x == context.pos_x + context.dim_x - 1:
            # store
            if begin_edge == 1:
                index = begin
            elif end_edge == 1:
                index = end
            context.polygons[index] = description
        elif (edge_sum & 0x4) != 0 and y == context.pos_y + context.dim_y - 1:
            # store
            if begin_edge == 2:
                index = begin
            elif end_edge == 2:
                index = end
            context.polygons[index] = description
        elif (edge_sum & 0x8) != 0 and x == context.pos_x:
            # store
            if begin_edge == 3:
                index = begin
            elif end_edge == 3:
                index = end
            context.polygons[index] = description

        description_left = NULL
        description_top = NULL

        if (edge_sum & 0x2) != 0:
            # TODO: The previous comntent of y_prev have to be stored somewhere in case
            description_left = context.y_prev
            context.y_prev = description

        if (edge_sum & 0x4) != 0:
            # TODO: The previous comntent of x_scan have to be stored somewhere in case
            description_top = context.x_scan[x - context.pos_x]
            context.x_scan[x - context.pos_x] = description

        if description_left != NULL or description_top != NULL:
            if description_left == description_top:
                self._store_unconnected_polygon(context, description_left)
            else:
                if description_left != NULL:
                    self._store_unconnected_polygon(context, description_left)
                if description_top != NULL:
                    self._store_unconnected_polygon(context, description_top)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _replace_from_cache(self,
                                  TileContext_t *context,
                                  polygon_description_t *previous_description,
                                  polygon_description_t *new_description) nogil:
        if context.x_scan[previous_description.begin_x  - context.pos_x] == previous_description:
            context.x_scan[previous_description.begin_x - context.pos_x] = new_description
        if context.x_scan[previous_description.end_x - context.pos_x] == previous_description:
            context.x_scan[previous_description.end_x - context.pos_x] = new_description

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _store_unconnected_polygon(self, TileContext_t *context, polygon_description_t *description) nogil:
        cdef:
            int i
            map[hash_index_t, polygon_description_t*].iterator it_begin
            map[hash_index_t, polygon_description_t*].iterator it_end
        # still on the y cache
        if context.y_prev == description:
            return
        if context.y_prev2 == description:
            return

        # already stored as polygon to merge
        it_begin = context.polygons.find(description.begin)
        if it_begin != context.polygons.end():
            return
        it_end = context.polygons.find(description.end)
        if it_end != context.polygons.end():
            return

        # still on the x cache
        if context.x_scan[description.begin_x - context.pos_x] == description:
            return
        if context.x_scan[description.end_x - context.pos_x] == description:
            return

        # the polygon have to be stored as is
        context.final_polygons.push_back(description)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _store_unconnected_polygons(self, TileContext_t *context) nogil:
        cdef:
            int i
            polygon_description_t *description
        description = context.y_prev
        if description != NULL:
            context.y_prev = NULL
            self._store_unconnected_polygon(context, description)
        for i in range(context.dim_x):
            description = context.x_scan[i]
            if description != NULL:
                self._store_unconnected_polygon(context, description)
                self._replace_from_cache(context, description, NULL)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _merge_context(self, TileContext_t *context, TileContext_t *other) nogil:
        cdef:
            map[hash_index_t, polygon_description_t*].iterator it_begin
            map[hash_index_t, polygon_description_t*].iterator it_end
            map[hash_index_t, polygon_description_t*].iterator it
            polygon_description_t *description_other
            polygon_description_t *description
            polygon_description_t *description2
            hash_index_t vhash
            vector[polygon_description_t*] mergeable_polygons
            int i

        # merge final polygons
        context.final_polygons.splice(context.final_polygons.end(), other.final_polygons)

        # mergeable_polygons.reserve(other.polygons.size() / 2)
        it = other.polygons.begin()
        while it != other.polygons.end():
            vhash = dereference(it).first
            description_other = dereference(it).second
            if description_other.begin == vhash:
                mergeable_polygons.push_back(description_other)
            preincrement(it)

        for i in range(mergeable_polygons.size()):
            description_other = mergeable_polygons[i]
            it_begin = context.polygons.find(description_other.begin)
            it_end = context.polygons.find(description_other.end)

            if it_begin == context.polygons.end() and it_end == context.polygons.end():
                # It's a new polygon
                context.polygons[description_other.begin] = description_other
                context.polygons[description_other.end] = description_other
            elif it_end == context.polygons.end():
                # The head of the polygon have to be merged
                description = dereference(it_begin).second
                context.polygons.erase(description.begin)
                context.polygons.erase(description.end)
                if description.begin == description_other.begin:
                    description.begin = description.end
                    description.points.reverse()
                description.end = description_other.end
                # remove the dup element
                description_other.points.pop_front()
                description.points.splice(description.points.end(), description_other.points)
                context.polygons[description.begin] = description
                context.polygons[description.end] = description
                del description_other
            elif it_begin == context.polygons.end():
                # The tail of the polygon have to be merged
                description = dereference(it_end).second
                context.polygons.erase(description.begin)
                context.polygons.erase(description.end)
                if description.begin == description_other.end:
                    description.begin = description.end
                    description.points.reverse()
                description.end = description_other.begin
                description_other.points.reverse()
                # remove the dup element
                description_other.points.pop_front()
                description.points.splice(description.points.end(), description_other.points)
                context.polygons[description.begin] = description
                context.polygons[description.end] = description
                del description_other
            else:
                # Both sides have to be merged
                description = dereference(it_begin).second
                description2 = dereference(it_end).second
                if description == description2:
                    # It became a closed polygon
                    context.polygons.erase(description.begin)
                    context.polygons.erase(description.end)
                    if description.begin == description_other.begin:
                        description.begin = description.end
                        description.points.reverse()
                    description.end = description_other.end
                    # remove the dup element
                    description_other.points.pop_front()
                    description.points.splice(description.points.end(), description_other.points)
                    context.final_polygons.push_back(description)
                    del description_other
                else:
                    context.polygons.erase(description.begin)
                    context.polygons.erase(description.end)
                    context.polygons.erase(description2.begin)
                    context.polygons.erase(description2.end)
                    if description.begin == description_other.begin:
                        description.begin = description.end
                        description.points.reverse()
                    if description2.end == description_other.end:
                        description.end = description2.begin
                        description2.points.reverse()
                    else:
                        description.end = description2.end
                    description_other.points.pop_front()
                    description2.points.pop_front()
                    description.points.splice(description.points.end(), description_other.points)
                    description.points.splice(description.points.end(), description2.points)
                    context.polygons[description.begin] = description
                    context.polygons[description.end] = description
                    del description_other
                    del description2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _compute_point(self,
                             cnumpy.uint_t x,
                             cnumpy.uint_t y,
                             cnumpy.uint8_t edge,
                             cnumpy.float64_t isovalue,
                             point_t &result_point) nogil:
        cdef:
            int dx1, dy1, index1
            int dx2, dy2, index2
            cnumpy.float64_t fx, fy, ff, weight1, weight2
        # Use these to look up the relative positions of the pixels to interpolate
        dx1, dy1 = EDGE_TO_POINT[edge][0], EDGE_TO_POINT[edge][1]
        dx2, dy2 = EDGE_TO_POINT[edge + 1][0], EDGE_TO_POINT[edge + 1][1]
        # Define "strength" of each corner of the cube that we need
        index1 = (y + dy1) * self._dim_x + x + dx1
        index2 = (y + dy2) * self._dim_x + x + dx2
        weight1 = 1.0 / (EPSILON + fabs(self._image_ptr[index1] - isovalue))
        weight2 = 1.0 / (EPSILON + fabs(self._image_ptr[index2] - isovalue))
        # Apply a kind of center-of-mass method
        fx, fy, ff = 0.0, 0.0, 0.0
        fx += dx1 * weight1
        fy += dy1 * weight1
        ff += weight1
        fx += dx2 * weight2
        fy += dy2 * weight2
        ff += weight2
        fx /= ff
        fy /= ff
        result_point.x = x + fx
        result_point.y = y + fy

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef _extract_polygons(self):
        cdef:
            int i, i_pixel
            cnumpy.uint8_t index
            map[hash_index_t, polygon_description_t*].iterator it
            vector[polygon_description_t*] descriptions
            clist[point_t].iterator it_points
            polygon_description_t *description

        it = self._final_context.polygons.begin()
        while it != self._final_context.polygons.end():
            vhash = dereference(it).first
            description = dereference(it).second
            preincrement(it)

        with nogil:
            it = self._final_context.polygons.begin()
            while it != self._final_context.polygons.end():
                description = dereference(it).second
                if dereference(it).first == description.begin:
                    # polygones are stored 2 times
                    # only use one
                    descriptions.push_back(description)
                preincrement(it)
            self._final_context.polygons.clear()

            descriptions.insert(descriptions.end(),
                                self._final_context.final_polygons.begin(),
                                self._final_context.final_polygons.end())
            self._final_context.final_polygons.clear()

        # create result and clean up allocated memory
        polygons = []
        for i in range(descriptions.size()):
            description = descriptions[i]
            polygon = numpy.empty(description.points.size() * 2, dtype=numpy.float32)
            it_points = description.points.begin()
            i_pixel = 0
            while it_points != description.points.end():
                polygon[i_pixel + 0] = dereference(it_points).x
                polygon[i_pixel + 1] = dereference(it_points).y
                i_pixel += 2
                preincrement(it_points)
            polygon.shape = -1, 2
            polygons.append(polygon)
            del description
        return polygons

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def iso_contour(self, value=None):
        self._marching_squares(value)
        polygons = self._extract_polygons()
        return polygons
