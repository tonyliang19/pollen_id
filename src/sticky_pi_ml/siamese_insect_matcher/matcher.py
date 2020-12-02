import logging
import numpy as np
import networkx as nx
from sticky_pi_ml.siamese_insect_matcher.predictor import Predictor
from sticky_pi_ml.siamese_insect_matcher.ml_bundle import MLBundle
from sticky_pi_ml.tuboid import Tuboid
from sticky_pi_ml.image import ImageSeries, Image
import torch
from functools import lru_cache
from typing import List


class Matcher(object):
    _tub_min_length = 3

    def __init__(self, annotated_images_series: ImageSeries, ml_bundle: MLBundle, PredictorClass=Predictor):
        assert len(annotated_images_series) > 2
        self._annotated_images_series = annotated_images_series
        self._predictor = PredictorClass(ml_bundle)

    def __enter__(self):
        return self

    def __call__(self) -> List[Tuboid]:
        tuboids = self._draft_graph()
        logging.info('%i Stitching non-contiguous tuboids head to tail' % len(tuboids))
        tuboids = self._stitch_sub_graphs(tuboids)

        logging.info('%i tuboid detected. Merging conjoint tuboids' % len(tuboids))
        tuboids = self._merge_conjoint_tuboids(tuboids)

        logging.info('%i tuboid left. Removing small tuboid' % len(tuboids))
        tuboids = [tub for tub in tuboids if len(tub) > self._tub_min_length]

        logging.info('%i tuboid left. Cleaning up' % len(tuboids))
        # reordering tuboids
        Tuboid.set_instances_id(tuboids)
        return tuboids

    def _draft_graph(self) -> List[Tuboid]:
        dg = nx.DiGraph()
        annotated_images = self._annotated_images_series
        annotated_images.sort(key=lambda x: x.datetime)
        s0, s1 = annotated_images[0:2]
        for i, _ in enumerate(annotated_images[1:-1]):
            logging.info('Drafting, %i/%i' % (i + 1, len(annotated_images) - 2))
            s1 = annotated_images[i + 1]
            edges, nodes = self._predictor.match_two_images(s0, s1)

            # add the resulting nodes and edges to the overall graph
            # we define these primary edges as not stitched (as opposed to edges that span multiple frames)
            dg.add_weighted_edges_from(edges, stitched=False)
            # only add nodes that do not exist
            for k, v in nodes.items():
                if k not in dg.nodes:
                    dg.add_node(k)
                dg.nodes[k].update({'annotation': v})
            # optimisation: the first image had been cached so that
            # matching / extraction of the sub-image can be faster. For now, we clear the cache
            s0.clear_cache()
            s0 = s1

        s1.clear_cache()

        tuboids = []
        for sub_graph in nx.weakly_connected_components(dg):
            annotations = [nd[1]['annotation'] for nd in dg.subgraph(sub_graph).nodes(data=True)]
            tuboids.append(Tuboid(annotations, parent_series=self._annotated_images_series))
        return tuboids

    def _stitch_sub_graphs(self, tuboids: List[Tuboid]) -> List[Tuboid]:
        ntub = len(tuboids)
        if ntub < 2:
            return tuboids
        arr = np.zeros((ntub, ntub), dtype=np.float)
        for i in range(ntub):
            for j in range(i + 1, ntub):
                score = self._predictor.match_two_annots(tuboids[i].tail,
                                                         tuboids[j].head)
                arr[i, j] = score

        edges = []
        while np.sum(arr) > 0.0:
            i, j = np.unravel_index(arr.argmax(), arr.shape)
            edges.append((i, j, arr[i, j]))
            arr[i, :] = 0
            arr[:, j] = 0

        dg = nx.DiGraph()
        dg.add_weighted_edges_from(edges, stitched=True)

        for sdg in nx.weakly_connected_components(dg):
            annots_to_merge = []
            for tb in sdg:
                annots_to_merge.extend(tuboids[tb])
                # place holder to later delete without changing the indices yet
                tuboids[tb] = None
            merged_tuboid = Tuboid(annots_to_merge, parent_series=self._annotated_images_series)

            tuboids.append(merged_tuboid)
        tuboids = [tb for tb in tuboids if tb is not None]
        return tuboids

    # to tuboids are `conjoint' iff:
    # * their range overlap in time
    # * they have no coincident reads
    # Typically, conjoint tuboids are the same instance, but falsely clustered as multiple ones.
    # the goal is to iteratively merge conjoint tuboids
    def _merge_conjoint_tuboids(self, tuboids, iteration=0):
        ntub = len(tuboids)
        if ntub < 2:
            return tuboids
        logging.info('Merging conjoint tuboids. Iteration %i' % iteration)
        arr = np.zeros((ntub, ntub), dtype=np.float)
        for i in range(ntub):
            tb0 = tuboids[i]
            for j in range(i + 1, ntub):
                tb1 = tuboids[j]
                if self._is_tuboid_pair_conjoint(tb0, tb1):
                    match = self._predictor.match_two_tuboids(tb0, tb1)
                    if match:
                        arr[i, j] = match
        logging.info('n tuboid matches: %i' % np.sum(arr > 0))
        if np.sum(arr) == 0:
            return tuboids

        i, j = np.unravel_index(arr.argmax(), arr.shape)

        assert j > i
        tb1 = tuboids.pop(j)
        tb0 = tuboids.pop(i)

        merged_tuboid = Tuboid(tb1 + tb0, parent_series=self._annotated_images_series)
        tuboids.append(merged_tuboid)

        # we recurse as still need to merge some tuboids
        tuboids = self._merge_conjoint_tuboids(tuboids, iteration=iteration + 1)
        return tuboids

    @lru_cache(maxsize=None)
    def _is_tuboid_pair_conjoint(self, tb0, tb1):
        if len(tb0) + len(tb1) < 3:
            return False

        if not (tb0.tail_datetime > tb1.head_datetime > tb0.head_datetime or
                tb1.tail_datetime > tb0.head_datetime > tb1.head_datetime):
            return False

        # todo optimise. lists are presorted!
        for t0 in tb0:
            for t1 in tb1:
                if t1.datetime == t0.datetime:
                    return False
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._clean_up()

    def __del__(self):
        self._clean_up()

    def _clean_up(self):
        logging.info('deleting img cache!')
        for im in self._annotated_images_series:
            im.clear_cache()

    #
    # def draw_graph(self):
    #     import plotly.graph_objs as go
    #     pos = {}
    #     Xn, Yn, Zn = [], [], []
    #     labels = []
    #     for i, attrs in list(self._dg.nodes(data=True)):
    #         an = attrs['annotation']
    #         z= an.datetime.timestamp()
    #         center = an.center
    #         x, y = center.real, center.imag
    #         pos[i] = (x,y,z)
    #
    #         Xn += [x]  # x-coordinates of nodes
    #         Yn += [y]
    #         Zn += [z]
    #         if'tuboid_id' in attrs:
    #             tuboid_id = str(attrs['tuboid_id'])
    #         else:
    #             tuboid_id = ""
    #         labels.append('%s | %s | %f,%f, %i'% (tuboid_id, i, x,y,z))
    #
    #
    #     Xe = []
    #     Ye = []
    #     Ze = []
    #     edge_labels = []
    #     edge_width = []
    #     edge_color = []
    #
    #     for i in self._dg.edges(data=True):
    #         x0, y0, z0 = pos[i[0]]
    #         x1, y1, z1 = pos[i[1]]
    #         Xe += [x0, x1, None]  # x-coordinates of edge ends
    #         Ye += [y0, y1, None]
    #         Ze += [z0, z1, None]
    #         edge_width.append(i[2]['weight'] * 2)
    #         if i[2]['stitched']:
    #             edge_color += ['#ff0000'] *3
    #         else:
    #             edge_color += ['#0000ff'] *3
    #         edge_labels += ['%s -> %s | w=%f'% (i[0], i[1], i[2]['weight'])] * 3
    #
    #     trace1 = go.Scatter3d(x=Xe,
    #                           y=Ye,
    #                           z=Ze,
    #                           mode='lines',
    #                           line= dict(width = 5, color = edge_color),
    #                           # color = dict(color = edge_color),
    #                           hoverinfo='text',
    #                           text=edge_labels
    #                           )
    #
    #     trace2 = go.Scatter3d(x=Xn,
    #                           y=Yn,
    #                           z=Zn,
    #                           mode='markers',
    #                           name='insects',
    #                           marker=dict(symbol='circle',
    #                                       size=6,
    #                                       colorscale='Viridis'
    #                                       ),
    #                           hoverinfo='text',
    #                           text=labels
    #                           )
    #
    #     axis = dict(showbackground=False,
    #                 showline=False,
    #                 zeroline=False,
    #                 showgrid=False,
    #                 showticklabels=False,
    #                 title=''
    #                 )
    #
    #     layout = go.Layout(
    #         title="Tuboids",
    #         width=1000,
    #         height=1000,
    #         showlegend=False,
    #         scene=dict(
    #             xaxis=dict(axis),
    #             yaxis=dict(axis),
    #             zaxis=dict(axis),
    #         ),
    #         margin=dict(
    #             t=100
    #         ),
    #         hovermode='closest',
    #         annotations=[
    #             dict(
    #                 showarrow=False,
    #                 xref='paper',
    #                 yref='paper',
    #                 x=0,
    #                 y=0.1,
    #                 xanchor='left',
    #                 yanchor='bottom',
    #                 font=dict(
    #                     size=14
    #                 )
    #             )
    #         ], )
    #
    #     data=[trace1, trace2]
    #     fig=go.Figure(data=data, layout=layout)
    #     fig.show()

    def make_video(self, tuboids, out, scale=(1600, 1200), fps=4, show=False):
        import cv2

        def col_lut(id):
            np.random.seed(id)
            gr = np.round(np.random.random(2) * 255).astype(int).tolist()
            return 255, *gr

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(out, fourcc, fps, (scale[0], scale[1]))
        logging.info('Saving video in %s' % out)

        try:
            for i, s in enumerate(self._annotated_images_series):
                im = np.copy(s.read())
                logging.info('Processing %s (%i/%i)' % (s.filename, i, len(self._annotated_images_series)))
                for j, t in enumerate(tuboids):
                    assert s.device == t.device
                    bbox, inferred = t.bbox_at_datetime(s.datetime)
                    if bbox is not None:
                        col = col_lut(j)
                        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=col,
                                      thickness=3)
                        text = "%03d" % t.id
                        cv2.putText(im, text, (bbox[0] + bbox[2], bbox[1] + bbox[3]), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    color=col, thickness=3)

                cv2.putText(im, s.filename, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), thickness=3)

                im = cv2.resize(im, scale)
                if show:
                    cv2.imshow('im', im)
                    cv2.waitKey(250)
                vw.write(im)
        finally:
            if vw is not None:
                vw.release()
