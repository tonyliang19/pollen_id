import numpy as np
from networkx import DiGraph
from sticky_pi_ml.utils import pad_to_square
from typing import List
class Tuboid(object):
    def __init__(self, sub_directed_graph: DiGraph):
        self._sdg = sub_directed_graph
        nodes = sorted(list(sub_directed_graph.nodes()))
        self._head = {'name': nodes[+0], 'annotation': sub_directed_graph.nodes[nodes[+0]]['annotation']}
        self._tail = {'name': nodes[-1], 'annotation': sub_directed_graph.nodes[nodes[-1]]['annotation']}

        d = None
        for a in self.annotations:
            if d is not None:
                assert a.device == d
            else:
                d = a.device

        self._device = d
        self._id = -1
        self._all_timestamps = [a.datetime.timestamp() for a in self.annotations]
        self._all_timestamps = np.array(self._all_timestamps).astype(np.float)
        self._all_bboxes = np.array([a.bbox for a in self.annotations]).astype(np.float)
        self._cached_conv = {}

    @classmethod
    def set_instances_id(cls, tuboids: List[Tuboid]):
        tuboids.sort(key=lambda x: x.head_datetime)
        for i, tb in enumerate(tuboids):
            tb.set_id(i)

    @property
    def subgraph(self):
        return self._sdg

    @property
    def id(self):
        return self._id


    @property
    def annotations(self):
        out = [nd[1]['annotation'] for nd in self.subgraph.nodes(data=True)]
        out.sort(key=lambda x: x.datetime)
        return out

    def all_annotation_sub_images(self, masked = True, scale_width = None):
        out = []
        parent_images = []
        scale_ratios = []
        centers = []
        for a in self.annotations:
            out.append(a.subimage(masked=masked))
            parent_images.append(a.parent_image)
            centers.append(a.center)
        if scale_width is None:
            return zip(out, parent_images, [1] * len(out))
        for i, o in enumerate(out):
            scale_ratios.append(scale_width / max(out[i].shape[0:2]))
            out[i] = pad_to_square(o,  scale_width)

        return zip(out, parent_images, scale_ratios, centers)

    def __len__(self):
        return len(self._sdg.nodes)

    def __iter__(self):
        for a in self.annotations:
            yield a

    def __getitem__(self, item):
        return self.annotations[item]

    def set_id(self, i):
        self._id = i
        for n in self.subgraph.nodes(data=True):
            n[1].update({'tuboid_id': i})

    def bbox_at_datetime(self, datetime):
        if datetime < self.head_datetime or datetime > self.tail_datetime:
            return None, None

        x = np.array([datetime.timestamp()]).astype(np.float)
        if datetime in self._all_timestamps:
            inferred = False
        else:
            inferred = True

        out = []
        for i in range(4):
            out.append(int(np.interp(x, self._all_timestamps, self._all_bboxes[:, i])))

        return out, inferred



    @property
    def device(self):
        return self._device

    @property
    def tail(self):
        return self._tail

    @property
    def tail_datetime(self):
        return self._tail['annotation'].datetime

    @property
    def head_datetime(self):
        return self._head['annotation'].datetime

    @property
    def head(self):
        return self._head

    def set_cached_conv(self, hash, array):
        self._cached_conv[hash] = array

    def clear_cached_conv(self):
        self._cached_conv = {}
    @property
    def cached_conv(self):
        return self._cached_conv