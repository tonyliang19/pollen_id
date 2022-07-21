import logging

import cv2
import numpy as np
from shapely.geometry import Polygon


class Annotation(object):
    _fill_opacity = 0.2

    def __init__(self, contour, stroke_colour, parent_image = None, fill_colour='#ff0000', name='annotation', value=0, debug_info=None):
        self._contour = contour
        self._name = name
        self._stroke_colour = stroke_colour
        self._value = value
        self._fill_colour = fill_colour
        self._parent_image = parent_image

        self._bbox = cv2.boundingRect(contour)
        try:
            mom = cv2.moments(self._contour)
            cX = mom["m10"] / mom["m00"]
            cY = mom["m01"] / mom["m00"]

            self._center = cX + 01j * cY
        except ZeroDivisionError:
            if parent_image is not None:
                logging.warning('Division by zero in contour moment of image %s (%s).\n%s' %
                                (parent_image.filename, debug_info, str(self._contour)))
            else:
                logging.warning('Division by zero in contour moment')

        self._cached_conv = {}
        self._area = cv2.contourArea(contour)

    def to_dict(self):
        out =      {'contour': self._contour.tolist(),
                    'name': self._name,
                    'stroke_colour': self._stroke_colour,
                    'value': self._value,
                    'fill_colour': self._fill_colour}
        return out


    def rot_rect_width(self):
        try:
            rect = cv2.minAreaRect(self.contour)
            (_, _), (width, height), _ = rect
        except Exception as e:
            logging.warning(e)
            width, height = 0, 0

        if width < height:
            width, height = height, width
        return width


    def parent_image_array(self, cache=False):
        return self._parent_image.read(cache)

    @property
    def parent_image(self):
        return self._parent_image


    @property
    def stroke_col(self):
        return self._stroke_colour
    
    
    @property
    def fill_col(self):
        return self._fill_colour

    @property
    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    @property
    def value(self):
        return self._value

    @property
    def bbox(self):
        return self._bbox

    @property
    def contour(self):
        return self._contour

    @property
    def polygon(self):
        return Polygon(np.squeeze(self._contour))


    @property
    def area(self):
        return self._area

    @property
    def center(self):
        return self._center

    def set_cached_conv(self, hash, array):
        self._cached_conv[hash] = array

    def clear_cached_conv(self):
        self._cached_conv = {}
    @property
    def cached_conv(self):
        return self._cached_conv


    def svg_element(self, offset=(0,0)):
        d_list = []
        for i in range(len(self._contour)):
            x, y = self._contour[i][0]
            x += offset[0]
            y += offset[1]
            d_list.append(str(x) + ',' + str(y))
        d_str = ' '.join(d_list)
        out = '<path name="%s" value="%i" style="stroke:%s;stroke-opacity:1;fill:%s;fill-opacity:%f" d="M%s Z"/>' % \
              (self._name, self._value, self._stroke_colour, self._fill_colour, self._fill_opacity, d_str)
        return out


class DictAnnotation(Annotation):
    def __init__(self, dic, **kwargs):
        dic['contour'] = np.array(dic['contour'])
        super().__init__(**dic,**kwargs)
