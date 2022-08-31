from __future__ import annotations
from typing import Optional
from .sequence import Sequence, PulseUsage


class Distribution:
    def __init__(self, dist_type: str,
                 ancestor: Optional[tuple[str, Distribution]] = None) -> None:
        self.ancestors = [ancestor] if ancestor else []
        self.mag = None
        self.dist_type = dist_type
        self.kt_offset_vec = None
        self.prepass_rel_signal = 1.0 if self.dist_type == '+' else 0.0
        self.rel_influence = 1  # Dummy value that says this dist is important


def FID_graph(seq: Sequence) -> list[list[Distribution]]:
    """Create a graph that simulates z0 and one additional state."""
    graph = [[Distribution('z0')]]

    for pulse in map(lambda rep: rep.pulse, seq):
        rep = [Distribution('z0', ('zz', graph[-1][0]))]

        if pulse.usage == PulseUsage.EXCIT:
            rep.append(Distribution('+', ('z+', graph[-1][0])))
        elif len(graph[-1]) == 2:
            # Can't refocus if nothing is excited
            if pulse.usage == PulseUsage.REFOC:
                rep.append(Distribution('+', ('-+', graph[-1][1])))
            else:
                rep.append(Distribution('+', ('++', graph[-1][1])))

        # +z and -z are never used because there is no pulse.usage that flips
        # excited magnetisation back to z

        graph.append(rep)

    return graph
