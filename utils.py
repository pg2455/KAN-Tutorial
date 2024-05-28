import torch
import numpy as np


def extend_grid(grid, k):
    """
    Extends the grid on either size by k steps

    Args:
        grid: number of splines x number of control points
        k: spline order

    Returns:
        new_grid: number of splines x (number of control points + 2 * k)
    """
    n_intervals = grid.shape[-1] - 1
    bucket_size = (grid[:, -1] - grid[:, 0]) / n_intervals
    
    for i in range(k):
        grid = torch.cat([grid[:, :1] - bucket_size, grid], dim=-1)
        grid = torch.cat([grid, grid[:, -1:] + bucket_size], dim=-1)

    return grid


def eval_basis_functions(x_eval, grid, k):
    """
    Returns the value of basis functions defiend for order-k splines and control points defined in `grid`. 

    Args:
        x_eval: number of samples x number of dimensions
        grid: number of splines x number of control points
        k (scalar): order of spline

    Returns:
        bases: number of samples x number of dimensions x number of basis functions
    """
    grid_ = extend_grid(grid, k)
    # Reshape so that each x can be compared to each control point
    grid_ = grid_.unsqueeze(dim=2)
    x_ = x_eval.unsqueeze(dim=1)
    
    for idx in range(k+1):
        if idx == 0:
            bases = (x_ >= grid_[:, :-1]) * (x_ < grid_[:, 1:]) * 1.0 # step function; for each x in between the corresponding knots, the value is 1.
        else:
            bases1 = (x_ - grid_[:, :-(idx+1)]) / (grid_[:, 1:-idx] - grid_[:, :-(idx+1)]) * bases[:, :-1]
            bases2 = (grid_[:, (idx+1):] - x_) / (grid_[:, (idx+1):] - grid_[:, 1:-idx]) * bases[:, 1:]
            bases = bases1 + bases2

    return bases.transpose(1, 2) 


def get_coeff(bases, y_eval):
    """
    Returns coefficients that give y_eval from bases

    Args:
        bases: number of samples x number of basis functions
        y_eval: number of samples x 1
        
    """
    return torch.linalg.lstsq(bases.transpose(0, 1), y_eval.unsqueeze(dim=0)).solution


def single_stacked_kan_training(x_training, y_training, x_test, y_test, model_params=None,lr=0.1, k=2, n_layers=2, grid_sizes=[], grid_ranges= [], early_stopping_imrpovement_threshold=200, early_stopping_iterations=1e4, verbose=False, grid_range=[-1, 1], use_scales=False):
    """
    Trains a KAN of shape [1, 1, 1, ...1] with `n_layers` layers. 
    Args:
        x_training: Training inputs; number of samples x number of input dimensions
        y_training: Training targets; number of samples x 1
        x_test: Test inputs; number of samples x number of input dimensions
        y_test: Test targets; number of samples x 1
        model_params: Parameters of the model. Used in the Part 3 of the tutorial to continue training from an existing set of parameters. 
        lr: learning rate
        k: spline-order
        n_layers: number of layers in the KAN
        grid_sizes: Number of control points for each spline in the stack 
        grid_ranges: Grid ranges for each spline in the stack
        early_stopping_improvement_threshold: Number of iterations after which we can stop if there is no improvement in the validation loss
        early_stopping_iterations: Maximum number of iterations
        verbose: Whether to print the intermediate losses or not
        grid_range: Range of grids 
        use_scales: Whether to use the scaling parameters (see section 2. )
        
    """ 
    if grid_sizes == []:
        grid_sizes = [10] * n_layers

    if grid_ranges == []:
        grid_ranges = [[-1, 1]]* n_layers

    if not model_params:
        grids, coeffs, scale_bases, scale_splines, base_fns = [], [], [], [], []
        for idx in range(n_layers):
            grid = torch.linspace(grid_ranges[idx][0], grid_ranges[idx][1], steps=grid_sizes[idx]).unsqueeze(dim=0)
            grids.append(grid)
            
            coeff = torch.zeros((1, grid_sizes[idx] + k - 1, 1), requires_grad=True)
            coeffs.append(coeff)
    
            if use_scales:
                base_fn = torch.nn.SiLU()
                scale_base = torch.nn.Parameter(torch.ones(x_eval.shape[-1])).requires_grad_(True)
                scale_spline = torch.nn.Parameter(torch.ones(x_eval.shape[-1])).requires_grad_(True)
    
                scale_bases.append(scale_base)
                scale_splines.append(scale_spline)
                base_fns.append(base_fn)
    else:
        grids = model_params['grids']
        coeffs = model_params['coeffs']
        scale_bases = model_params['scale_bases']
        scale_splines = model_params['scale_splines']
        base_fns = model_params['base_fns']
    
    
    losses = {'train': [], 'val': []}
    best_loss = np.inf
    n_no_improvements = 0
    i = 0
    all_xs = []
    while True:    
        x = x_training
        xs = []
        for idx in range(n_layers):
            bases = eval_basis_functions(x, grids[idx], k)
            x_ = torch.einsum('ijk, bij->bk', coeffs[idx], bases)
            if use_scales:
                base_transformed_x = base_fns[idx](x) # transformation of the original x
                x = base_transformed_x * scale_bases[idx] + x_ * scale_splines[idx]
            else:
                x = x_

            xs.append(x.detach())

        all_xs.append(xs)
    
        y_pred = x
        loss = torch.mean(torch.pow(y_pred - y_training, 2))
        loss.backward()
        losses['train'].append(loss.item())

        # Gradient descent step
        for params in coeffs + scale_bases + scale_splines:
            params.data = params.data - lr * params.grad
            params.grad.zero_()

        # evaluate validation loss
        with torch.no_grad():
            x = x_test
            for idx in range(n_layers):
                bases = eval_basis_functions(x, grids[idx], k)
                x_ = torch.einsum('ijk, bij->bk', coeffs[idx], bases)
                if use_scales:
                    base_transformed_x = base_fns[idx](x) # transformation of the original x
                    x = base_transformed_x * scale_bases[idx] + x_ * scale_splines[idx]
                else:
                    x = x_
            y_pred_test = x
            val_loss = torch.mean(torch.pow(x - y_test, 2))
            
            losses['val'].append(val_loss.item())

        if i% 100 == 0 and verbose:
            print(f"Val loss: {val_loss.item(): 0.5f}\tTrain loss: {loss.item(): 0.5f}\tBest Val loss:{best_loss: 0.5f}")
            
        if best_loss > val_loss.item():
            best_loss = val_loss.item()
            best_model = (coeffs, base_fns, scale_bases, scale_splines)
            n_no_improvements = 0
        else:
            n_no_improvements += 1
            if n_no_improvements > early_stopping_imrpovement_threshold:
                print('Stopping: No further improvements...')
                break
    
        i += 1
        if i > early_stopping_iterations:
            print('Stopping: Iteration limit reached...')
            break       

    model_params = {
        'grids': grids,
        'coeffs': best_model[0],
        'scale_bases':  best_model[2],
        'scale_splines':  best_model[3],
        'base_fns': best_model[1],
    }
    return model_params, y_pred_test, losses, all_xs