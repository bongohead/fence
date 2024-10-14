""" 
Visualizations for FENCE
"""

import torch
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots

@torch.no_grad()
def visualize_fence(
    decoded_text: list[str], 
    hks: list[np.ndarray], 
    layers: list[int], 
    fence_dict: dict,
    start_dim: int | None = None,
    end_dim: int | None = None,
    min_range: float | int = 0, 
    max_range: float | int = 1
    ):
    """
    Graph an average of hidden state layer outputs

    Params:
        @decoded_text: A list of output strings
        @hks: The list of transformer layers outputs, each a numpy array of size N x D
        @layers: A list of transformer layer indices to graph. Indices start from 1, not 0.
        @fence_dict: The FENCE feature dictionary.
        @start_dim: The starting dimension (1-indexed) to display the graph on. If None, sets to the minimum dimension in fence_dict minus 20.
        @end_dim: The ending dimension (1-indexed) to display the graph on. If None, sets to the maximum dimension in fence_dict plus 20.
        @min_range: The minimum (red) FENCE value, or a list of values equal to the length of layers
        @max_range: The maximum (green) FENCE value, or a list of values equal to the length of layers
    """
    if (min_range >= max_range):
        raise ValueError('max_range must be greater than min_range!')

    if start_dim is None:
        start_dim = min(min(values) for values in fence_dict.values()) - 200
        start_dim = 1 if start_dim < 1 else start_dim

    if end_dim is None:
        end_dim = max(max(values) for values in fence_dict.values()) + 200
        end_dim = hks[0].shape[1] if end_dim > hks[0].shape[1] else end_dim

    subset_layers = [hks[i - 1] for i in layers]
    test_mat = np.mean(np.stack(subset_layers), axis = 0)
    draw_mat = (test_mat[:, (start_dim - 1):(end_dim)])
    
    custom_x = [x + 1 for x in list(range(start_dim - 1, test_mat.shape[1]))]

    filtered_x = [custom_x[i] for i in range(len(custom_x)) if (custom_x[i] % 10) == 0]
    filtered_indices = [i for i in range(len(custom_x)) if (custom_x[i] % 10)  == 0]
    
    custom_y = decoded_text[:-1]
    custom_colorscale = [
        [0, 'rgba(128, 128, 128, 0.9)'], [0.1999, 'rgba(128, 128, 128, 0.9)'], 
        [0.20, 'rgba(212, 72, 88, .9)'], [0.33, 'rgba(212, 72, 88, .9)'], [0.42, 'rgba(245, 125, 21, .5)'], 
        [0.50, 'rgba(250, 194, 40, .5)'],
        [0.60, 'rgba(181, 222, 43, .9)'], [0.68, 'rgba(94, 201, 98, .9)'], [0.80, 'rgba(94, 201, 98, .9)'],
        [0.8001, 'rgba(128, 128, 128, 0.9)'], [1.0, 'rgba(128, 128, 128, 0.9)']
    ]

    tol_scale_factor = 0.8 # Extra "tolerance" range around min/max where red/green is shown instead of gray
    
    fig = px.imshow(
        draw_mat, color_continuous_scale = custom_colorscale,
        labels = dict(x = 'Dimension', y = 'Token', color = 'Output'),
        zmin = min_range - (max_range - min_range) * tol_scale_factor, zmax = max_range + (max_range - min_range) * tol_scale_factor,
        aspect = 'auto'
        )\
        .update_layout(
            height = 10 + len(custom_y) * 12,
            plot_bgcolor = 'white', paper_bgcolor = 'white', margin = dict(l = 50, r = 0, t = 50, b = 20),
            coloraxis_colorbar = {'orientation': 'h', 'yanchor': 'top', 'xanchor': 'right', 'y': -0.1, 'x': 1.0, 'title': None, 'len': 0.2, 'thickness': 10, 'tickfont': {'size': 10}}
        )\
        .update_yaxes(tickvals = list(range(len(custom_y))), ticktext = custom_y, tickfont = {'size': 10.5})\
        .update_xaxes(tickvals = filtered_indices, ticktext = filtered_x, tickfont = {'size': 10.5})

    for fname, fdim in fence_dict.items():
        start_ix = fdim[0] - (start_dim - 1) - 1
        end_ix = fdim[1] - (start_dim - 1)
        fig\
            .add_shape(type = 'rect', x0 = start_ix - .5, y0 = -.5, x1 = end_ix - .5, y1 = test_mat.shape[0] - .5,  xref = 'x', yref = 'y', line = {'color': 'black', 'width': 2})\
            .add_annotation(
                x = (start_ix + end_ix - 1)/2, y = -2, xref = 'x', yref = 'y', 
                # text =  fname + ' (<span style="text-decoration:overline">D<sub>f</sub></span> = ' + str(test_mat[:, (start_ix - 1):end_ix].mean().round(2)) + ')',
                text =  fname,
                showarrow = False, font = {'color': 'black', 'size': 11}, align = 'center'
            )

    return fig