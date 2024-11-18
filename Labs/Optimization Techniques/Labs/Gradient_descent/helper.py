import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
# Draw model
def draw_model(data, model, theta, title=None):
    x_model = np.arange(0, 2, 0.01)
    y_model = model(theta, x_model)

    fig, ax = plt.subplots()
    ax.plot(data[0, :], data[1, :], 'bo')  # Plot the data points
    ax.plot(x_model, y_model, 'm-')        # Plot the model (line)
    
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    
    if title is not None:
        ax.set_title(title)
    
    plt.show()

def compute_loss(data_x, data_y, model, theta):
    """
    Computes Mean Squared Error (MSE) loss between predicted and true values.
    
    Parameters:
    data_x : array-like
        Input feature data (independent variable).
    data_y : array-like
        True output data (dependent variable).
    model : function
        The model that predicts `y` values based on `x` and `theta`.
    theta : array-like
        Model parameters (weights).
    
    Returns:
    loss : float
        The computed MSE loss.
    """
    
    # Step 1: Make predictions using the model
    pred_y = model(theta, data_x)
    
    # Step 2: Compute squared differences between predictions and actual values
    squared_diffs = (pred_y - data_y) ** 2
    
    # Step 3: Sum all squared differences and return as loss
    loss = np.sum(squared_diffs)
    
    return loss

def draw_loss_function(compute_loss, data, model, theta_iters=None):
    # Define pretty colormap
    my_colormap_vals_hex = ('2a0902', '2b0a03', '2c0b04', '2d0c05', '2e0c06', '2f0d07', '300d08', '310e09', 
                            '320f0a', '330f0b', '34100b', '35110c', '36110d', '37120e', '38120f', 
                            # Add more color values as needed...
                           )
    
    my_colormap_vals_dec = np.array([int(element, base=16) for element in my_colormap_vals_hex])
    r = np.floor(my_colormap_vals_dec / (256 * 256))
    g = np.floor((my_colormap_vals_dec - r * 256 * 256) / 256)
    b = np.floor(my_colormap_vals_dec - r * 256 * 256 - g * 256)
    my_colormap = ListedColormap(np.vstack((r, g, b)).transpose() / 255.0)

    # Make grid of intercept/slope values to plot
    intercepts_mesh, slopes_mesh = np.meshgrid(np.arange(0.0, 2.0, 0.02), np.arange(-1.0, 1.0, 0.002))
    loss_mesh = np.zeros_like(slopes_mesh)

    # Compute loss for every set of parameters
    for idslope, slope in np.ndenumerate(slopes_mesh):
        loss_mesh[idslope] = compute_loss(data[0, :], data[1, :], model,
                                          np.array([[intercepts_mesh[idslope]], [slope]]))

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    
    # Contour plot for loss function
    ax.contourf(intercepts_mesh, slopes_mesh, loss_mesh, 256, cmap=my_colormap)
    ax.contour(intercepts_mesh, slopes_mesh, loss_mesh, 40, colors=['#80808080'])

    # Plot theta iterations if provided
    if theta_iters is not None:
        ax.plot(theta_iters[0, :], theta_iters[1, :], 'go-')

    ax.set_ylim([1, -1])
    ax.set_xlabel('Intercept')
    ax.set_ylabel('Slope')
    
    plt.show()

def loss_function_1D(dist_prop, data, model, theta_start, search_direction):
    """
    Computes the loss after moving a certain distance along the search direction.
    
    Parameters:
    - dist_prop: The proportion of the search direction to move.
    - data: The dataset (x and y values).
    - model: The model used to make predictions.
    - theta_start: The starting point of theta (parameters).
    - search_direction: The direction along which to search for the minimum loss.
    
    Returns:
    - The computed loss after moving along the search direction.
    """
    return compute_loss(data[0,:], data[1,:], model, theta_start + search_direction * dist_prop)

def line_search(data, model, theta, gradient, thresh=0.00001, max_dist=0.1, max_iter=15, verbose=False):
    """
    Performs a line search to find the optimal step size along the gradient direction.

    Parameters:
    - data: The dataset (x and y values).
    - model: The model used to make predictions.
    - theta: Current parameters (intercept and slope).
    - gradient: The gradient vector (direction of steepest descent).
    - thresh: Threshold for stopping criteria based on distance between points.
    - max_dist: Maximum distance to search along the gradient direction.
    - max_iter: Maximum number of iterations for line search.
    - verbose: If True, prints intermediate steps.

    Returns:
    - The optimal step size found by the line search.
    """
    
    # Initialize four points along the range we are going to search
    a = 0
    b = 0.33 * max_dist
    c = 0.66 * max_dist
    d = 1.0 * max_dist
    n_iter = 0

    # While we haven't found the minimum closely enough
    while np.abs(b - c) > thresh and n_iter < max_iter:
        # Increment iteration counter (just to prevent an infinite loop)
        n_iter += 1
        
        # Calculate all four points
        lossa = loss_function_1D(a, data, model, theta, gradient)
        lossb = loss_function_1D(b, data, model, theta, gradient)
        lossc = loss_function_1D(c, data, model, theta, gradient)
        lossd = loss_function_1D(d, data, model, theta, gradient)

        if verbose:
            print(f'Iter {n_iter}, a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}')
            print(f'a {lossa:.6f}, b {lossb:.6f}, c {lossc:.6f}, d {lossd:.6f}')

        # Rule #1 If point A is less than points B, C, and D then halve distance from A to points B,C,D
        if np.argmin((lossa, lossb, lossc, lossd)) == 0:
            b = a + (b - a) / 2
            c = a + (c - a) / 2
            d = a + (d - a) / 2
            continue

        # Rule #2 If point B is less than point C then:
        #                     D becomes C,
        #                     B becomes 1/3 between A and new D,
        #                     C becomes 2/3 between A and new D
        if lossb < lossc:
            d = c
            b = a + (d - a) / 3
            c = a + 2 * (d - a) / 3
            continue

        # Rule #3 If point C is less than point B then:
        #                     A becomes B,
        #                     B becomes 1/3 between new A and D,
        #                     C becomes 2/3 between new A and D
        a = b
        b = a + (d - a) / 3
        c = a + 2 * (d - a) / 3

    # Return average of two middle points as optimal step size
    return (b + c) / 2.0