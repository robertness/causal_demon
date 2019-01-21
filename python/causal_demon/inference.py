import pyro


def infer_dist(prog, n_dist):
    """Obtain the unique distribution entailed by a SCM program.

    Do inference on a SCM program and obtain a object representing the
    probability distribution entailed by the SCM.

    This implementation depends on simple importance sampling with 5000
    samples.

    TODO(Kuashal): Do an implementation with stochastic variational inference.
    http://pyro.ai/examples/svi_part_i.html


    `prog`: the subroutine encoding the SCM.
    `n_dist`: a dictionary containing distributions for each
    noise object.
    """
    return pyro.infer.Importance(prog, num_samples=5000).run(n_dist)
