class Palette(object):
    def __init__(self, dic: dict):

        """
        A class to map annotation stroke colour to ML label.

        :param dic: a mapping of label-> stroke. e.g. {'candidate': '#ffff00', 'drosophila':'#ff0000'}
        """
        self._class_name_to_stroke_map = {}
        self._stroke_to_class_name_map = {}
        self._id_map = {}
        for i,k in enumerate(sorted(dic)):
            self._class_name_to_stroke_map[k] = {'id': i + 1,
                                                 'stroke': dic[k]}

            self._stroke_to_class_name_map[dic[k]] = {'id': i + 1,
                                                 'class': k}

            self._id_map[i+1] = {'class': k,
                                                      'stroke': dic[k]}
    @property
    def classes(self):
        return [v['class'] for v in self._id_map.values()]

    def class_to_id(self, exclude_classes = ()):
        out = {}
        for k, v in self._class_name_to_stroke_map.items():
            if k not in exclude_classes:
                out[k] = v['id']
        return out

    def to_gimp_palette(self, target, name=None):
        if not name:
            name = self.__class__.__name__

        with open(target, 'w') as f:
            f.write("GIMP Palette\n")
            f.write("Name: %s\n" % name)
            f.write("#\n")
            for k, v in self._class_name_to_stroke_map.items():
                h = v['stroke'].lstrip('#')
                rgb = [str(int(h[i:i + 2], 16)) for i in (0, 2, 4)]
                rgb = "\t".join(rgb)
                line = "%s\t%s (%s)\n" % (rgb, k, v['stroke'])
                f.write(line
                        )

    def get_stroke_from_id(self, id):
        return self._id_map[id]['stroke']


    def get_stroke_from_class(self, cls):
        return self._class_name_to_stroke_map[cls]['stroke']

    def get_class_from_id(self, id):
        return self._id_map[id]['class']

    def get_id_annot(self, annotation):
        col = annotation.stroke_col
        if col not in self._stroke_to_class_name_map:
            raise Exception('Could not find the colour %s in the ML palette. Classes %s' %
                            (col, str(self._stroke_to_class_name_map)))
        return self._stroke_to_class_name_map[col]['id']
