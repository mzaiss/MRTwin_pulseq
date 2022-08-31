from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from . import util
from .sequence import Sequence


def visualize(graph: list[list], seq: Sequence) -> list[np.ndarray]:
    """Plot the states every event of a sequence to an array of images."""
    events = []

    for dists, rep in zip(graph[1:], seq):
        for dist in dists:
            dist.kt_offset_vec = dist.prepass_kt_vec

        for e in range(rep.event_count):
            states = {
                '+': {'x': [], 'y': [], 's': []},
                'z': {'x': [], 'y': [], 's': []},
                'z0': {'x': [], 'y': [], 's': []},
            }
            events.append(states)

            for dist in dists:
                if dist.dist_type == '+':
                    dist.kt_offset_vec[0] += rep.gradm[e, 0].item()
                    dist.kt_offset_vec[1] += rep.gradm[e, 1].item()
                states[dist.dist_type]['x'].append(dist.kt_offset_vec[0])
                states[dist.dist_type]['y'].append(dist.kt_offset_vec[1])
                states[dist.dist_type]['s'].append(dist.rel_influence * 50)

    # Extends
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    for ev in events:
        x_min = min(x_min, *ev['+']['x'], *ev['z']['x'], *ev['z0']['x'])
        x_max = max(x_max, *ev['+']['x'], *ev['z']['x'], *ev['z0']['x'])
        y_min = min(y_min, *ev['+']['y'], *ev['z']['y'], *ev['z0']['y'])
        y_max = max(y_max, *ev['+']['y'], *ev['z']['y'], *ev['z0']['y'])

    img_array = []

    for e, event in enumerate(events):
        print(f"\rPloting event {e+1} / {len(events)}", end="")
        plt.figure(figsize=(5, 5))
        plt.grid()
        plt.xlim(1.1 * x_min, 1.1 * x_max)
        plt.ylim(1.1 * y_min, 1.1 * y_max)
        plt.xlabel("gradm x")
        plt.ylabel("gradm y")

        plt.scatter(**event['+'], label='+')
        plt.scatter(**event['z'], label='z')
        plt.scatter(**event['z0'], label='z0')
        plt.legend()

        img_array.append(util.current_fig_as_img(180))
        plt.close()

    print(" - done")
    return img_array
