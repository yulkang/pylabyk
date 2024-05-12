#  Copyright (c) 2024  Yul HR Kang. hk2699 at caa dot columbia dot edu.

from pylabyk import numpytorch as npt


def main():
    v_state_dim0 = npt.tensor([0., 90., 180., 270.])
    # CAVEAT: Using period is not really useful
    #   unless the distribution is guaranteed to be
    #   unifmodal & have negligible dispersion
    #   (which are not being checked)

    for v_dim0 in v_state_dim0:
        for offset in [-0.1, 0., 0.1]:
            print('\n---')
            raise(NotImplementedError(
                'find_neighbors_p_v() is not implemented; '
                'Also, Using period is not really useful unless the distribution is guaranteed to be'
                'unimodal & have negligible dispersion (which are not being checked) '))

            npt.find_neighbors_p_v(
                v_dim=v_dim0 + offset,
                v_state_dim0=v_state_dim0,
                period=360.,
                verbose=True
            )


if __name__ == '__main__':
    main()
