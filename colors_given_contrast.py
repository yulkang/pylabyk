#  Copyright (c) 2022  Yul HR Kang. hk2699 at caa dot columbia dot edu.


import numpy as np
from pylabyk import rotateVectors, np2


GROUND = 0
CEILING = 1
WALLS = np.arange(2, 6)


def colors_given_contrast(
    contrast_btw_walls: float, contrast_ground: float, contrast_ceiling: float = None,
    wall0=(.6, .4, .5)
) -> np.ndarray:
    """
    Use right square bipyramid (dual of a square prism) in the color space
    to give colors of a given contrast.
    To fit within 3-dimensions (R, G, B), the contrasts must meet the following
    conditions:
        contrast_ground >= contrast_btw_walls  / sqrt(2)
        contrast_ceiling >= contrast_btw_walls / sqrt(2)
    :param contrast_btw_walls: contrast between neighboring walls
    :param contrast_ground: contrast between ground and walls
    :param contrast_ceiling: contrast between ceiling and walls.
        If None, set to contrast_ground.
    :param wall0: (R, G, B); R+G+B must equal 1.5
    :return: colors[(ground, ceiling), (R, G, B)] = component
        where 0 <= component <= 1.
    """
    eps = 1e-6

    if contrast_ceiling is None:
        contrast_ceiling = contrast_ground

    ground0 = np.r_[0., 0., 0.]
    ceiling0 = np.r_[1., 1., 1.]
    midpoint = np.r_[.5, .5, .5]
    wall0 = np.r_[wall0]

    # def check_wall(wall: np.ndarray):
    #     assert np.abs(np.dot(midpoint - ground0, wall - midpoint)) < eps
    #     assert np.all(wall >= 0.)
    #     assert np.all(wall <= 1.)

    walls0 = np.stack([
        rotateVectors.rotateAbout(wall0, midpoint, np2.deg2rad(deg))
        for deg in [0., 90., 180., 270.]
    ])
    # for wall in walls0:
    #     check_wall(wall)

    contrast_btw_walls0 = np.linalg.norm(walls0[1] - walls0[0])
    walls = (
        (walls0 - midpoint[None]) / contrast_btw_walls0 * contrast_btw_walls
        + midpoint[None]
    )
    radius = np.linalg.norm(walls[0] - midpoint)
    ground = (
        (ground0 - midpoint) / np.linalg.norm(ground0 - midpoint)
        * np.sqrt(contrast_ground ** 2 - radius ** 2)
        + midpoint
    )
    ceiling = (
        (ceiling0 - midpoint) / np.linalg.norm(ceiling0 - midpoint)
        * np.sqrt(contrast_ceiling ** 2 - radius ** 2)
        + midpoint
    )

    colors = np.concatenate([ground[None], ceiling[None], walls], 0)

    assert np.all(colors >= 0.)
    assert np.all(colors <= 1.)
    assert len(colors) == 6
    n_wall = len(colors) - 2
    for i_wall in range(n_wall):
        assert np.abs(
            np.linalg.norm(colors[i_wall + 2] - colors[GROUND])
            - contrast_ground
        ) < eps
        assert np.abs(
            np.linalg.norm(colors[i_wall + 2] - colors[CEILING])
            - contrast_ceiling
        ) < eps
        assert np.abs(
            np.linalg.norm(colors[i_wall + 2] - colors[(i_wall + 1) % 4 + 2])
            - contrast_btw_walls
        ) < eps

    return colors


def test_colors():
    for contrast_ground, contrast_ceiling, contrast_btw_walls in [
        (0.4, 0.6, 0.5),
        (0.6, 0.4, 0.5),
        (0.5, 0.5, 0.2)
    ]:
        colors = colors_given_contrast(
            contrast_ground=contrast_ground,
            contrast_ceiling=contrast_ceiling,
            contrast_btw_walls=contrast_btw_walls
        )


if __name__ == '__main__':
    test_colors()
