import numpy as np
import math
import collections

NEG_INF = -float("inf")  # 表示正无穷


def make_new_beam():
    fn = lambda: (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def decode(probs, beam_size=10, blank=1427):
    T, S = probs.shape
    # probs = np.log(probs + 1e-5)
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T):
        next_beam = make_new_beam()
        for s in range(S):
            p = probs[t, s]
            for prefix, (p_b, p_nb) in beam:
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    n_p_nb = logsumexp(n_p_nb, p_b + p)
                next_beam[n_prefix] = (n_p_b, n_p_nb)
                if s == end_t:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)

        beam = sorted(next_beam.items(), key=lambda x: logsumexp(*x[1]), reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -logsumexp(*best[1])


if __name__ == "__main__":
    np.random.seed(3)

    time = 200
    output_dim = 1428

    probs = np.random.rand(time, output_dim)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    labels, score = decode(probs)
    print("labels:%s score:%.3f" % (labels, score))
