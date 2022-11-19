

import numpy as np
from numpy.linalg import svd


def ten2tt(x, tt_shapes, tt_ranks):
    # tt_ranks_ = tt_ranks.copy()
    d = len(tt_shapes)
    t = x
    tt_cores = []
    for i in range(d - 1):
        t = np.reshape(t, [tt_ranks[i] * tt_shapes[i], -1])
        u, s, v = svd(t, full_matrices=False)
        if s.shape[0] < tt_ranks[i + 1]:
            tt_ranks[i + 1] = s.shape[0]

        u = u[:, :tt_ranks[i + 1]]
        s = s[:tt_ranks[i + 1]]
        v = v[:tt_ranks[i + 1], :]

        tt_cores.append(np.reshape(u, [tt_ranks[i], tt_shapes[i], tt_ranks[i + 1]]))
        t = np.dot(np.diag(s), v)
    t = np.reshape(t, [tt_ranks[d - 1], tt_shapes[d - 1], tt_ranks[d]])
    tt_cores.append(t)
    # print(tt_ranks)

    return tt_cores


def tt2ten(tt_cores, tt_shapes):
    d = len(tt_cores)
    t = tt_cores[0]
    for i in range(1, d):
        rank = tt_cores[i].shape[0]
        t = np.reshape(t, [-1, rank])
        t = np.dot(t, np.reshape(tt_cores[i], [rank, -1]))

    t = np.reshape(t, tt_shapes)
    return t
