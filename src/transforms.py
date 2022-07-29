import torch
import torch.nn.functional as F


def vec_to_perpective_matrix(vec) -> torch.Tensor:

    out = torch.cat((vec, torch.ones(
        (vec.shape[0], 1), dtype=vec.dtype, device=vec.device)), dim=1).reshape(vec.shape[0], -1)
    return out.reshape(-1, 3, 3)


def get_grids(shape, homogenous=False, device="cpu") -> torch.Tensor:
    N, C, H, W = shape
    y, x = torch.meshgrid([torch.linspace(-1, 1, H), torch.linspace(-1, 1, W)])
    x, y = x.flatten(), y.flatten()
    if homogenous:
        grid = torch.stack([x, y, torch.ones(x.shape[0])], dim=0)  # (1,3,H*W)
    else:
        grid = torch.stack([x, y], dim=0)  # (1,2,H*W)
    grid = grid.repeat(N, 1, 1)  # repeat for batch dim
    return grid.to(device)


def permute_grid_for_interpolation(grid, shape):
    N, C, H, W = shape
    # (N, H, W, 2); cf torch.functional.grid_sample
    grid = grid.permute(0, 2, 1).reshape(-1, H, W, 2)
    grid = grid.expand(N, *grid.shape[1:])  # expand to minibatch

    return grid

###############################################################################
# TRANSFORMATIONS
###############################################################################


def affine_warp(theta, shape, exp=True, grid=None, return_homogenous_grid=False, inverse=False, global_transform=None ,device="cpu") -> torch.Tensor:
    """
    Args:
        theta: (N, 6)
        shape: (N, C, H, W)
        grid: (N, 2, H*W)
        inverse: bool
        exp: use matrix exponental - when learning Lie Algebra via STN (bool)
    Returns:
        grid: transofrmed grid (N, 2, H*W)
        use grid.permute(0,2,1).view(-1, H, W, 2) to get back to image coordinates

    """
    N, C, H, W = shape

    if grid is None:
        grid = get_grids(shape, homogenous=True, device=device)
    elif grid.shape[1] == 2:
        # convert to homogenous - grid is [N, 2, H*W]
        x, y = grid[:, 0, :].reshape(N, -1), grid[:, 1, :].reshape(N, -1)
        x, y = x.reshape(N, -1), y.reshape(N, -1)
        grid = torch.stack(
            [x, y, torch.ones((N, x.shape[-1]),device=device)], dim=1)  # (N, 3, H*W)

    # reshape for matmul
    theta = theta.reshape(-1, 2, 3)  # (N,6) -> (N, 2, 3)
    theta = F.pad(theta, (0, 0, 0, 1))  # (N, 3, 3)

    if exp:
        theta = torch.matrix_exp(theta)
    else:
        theta[:, -1, -1] = 1  # last row [0,0,0] -> [0,0,1]

    if global_transform is not None:
        theta = theta@global_transform
    if inverse:
        theta = torch.inverse(theta)

    # transform grid
    # (N,3,3)matmul(N,3,H*W) -> (N,3,HW) with torch.matmul
    grid_t = theta.matmul(grid)

    if not return_homogenous_grid:
        # convert to inhomogeneous coords (N, 2, HW)
        grid_t = grid_t[:, :2, :]

    return theta, grid_t


def homography_warp(theta, shape, exp=True, grid=None, return_homogenous_grid=False, inverse=False, device="cpu") -> torch.Tensor:
    """
    Args:
        theta: (N, 8)
        shape: (N, C, H, W)
        grid: (N, 2, H*W)
        inverse: bool
        exp: use matrix exponental - when learning Lie Algebra via STN (bool)
    Returns:
        grid: transofrmed grid (N, 2, H*W)

    """
    assert len(shape) == 4, "Shape must be 4D: N, C, H, W"
    N, C, H, W = shape
    theta = vec_to_perpective_matrix(theta)
    if exp:
        theta[:, 2, 2] = -theta[:, 0, 0]-theta[:, 1, 1]  # theta[2,2] = -a-e
        theta = torch.matrix_exp(theta)

    if inverse:
        theta = torch.inverse(theta)

    if grid is None:
        grid = get_grids(shape, homogenous=True, device=device)
    elif grid.shape[1] == 2:
        # convert to homogenous - grid is [N, 2, H*W]
        x, y = grid[:, 0, :].reshape(N, -1), grid[:, 1, :].reshape(N, -1)
        x, y = x.reshape(N, -1), y.reshape(N, -1)
        grid = torch.stack(
            [x, y, torch.ones((N, x.shape[-1]), device=device)], dim=1)  # (N, 3, H*W)
    # transform grid
    # (N,3,3)matmul(N,3,H*W) -> (N,3,HW) with torch.matmul
    grid_t = theta.matmul(grid)

    # convert to inhomogeneous coords -- cf Szeliski eq. 2.21
    grid_t = grid_t[:, :2, :] / (grid_t[:, 2, :].unsqueeze(1) + 1e-9)

    return theta, grid_t

