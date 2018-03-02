"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import math
import random
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, z1, y2, x2, z2]
    boxes: [boxes_count, (y1, x1, z1, y2, x2, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas

    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[3], boxes[:, 3])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[4], boxes[:, 4])
    z1 = np.maximum(box[2], boxes[:, 2])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 -z1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, z1, y2, x2, z2)]. Notice that (y2, x2, z2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    z1 = boxes[:, 2]
    y2 = boxes[:, 3]
    x2 = boxes[:, 4]
    z2 = boxes[:, 5]
    area = (y2 - y1) * (x2 - x1) * (z2 - z1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, z1, y2, x2, z2)]. Note that (y2, x2, z2) is outside the box.
    deltas: [N, (dy, dx, dz, log(dh), log(dw), log(dd))]
    """
    print ("apply bbox deltas")
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    print boxes.shape
    height = boxes[:, 3] - boxes[:, 0]
    width = boxes[:, 4] - boxes[:, 1]
    depth = boxes[:, 5] - boxes[:, 2]
    print ("apply bbox deltas")
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    center_z = boxes[:, 2] + 0.5 * depth
    print ("apply bbox deltas")
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    center_z += deltas[:, 2] * depth
    print ("apply bbox deltas")
    height *= np.exp(deltas[:, 3])
    width *= np.exp(deltas[:, 4])
    depth *= np.exp(deltas[:, 5])
    # Convert back to y1, x1, z1, y2, x2, z2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z1 = center_z - 0.5 * depth
    y2 = y1 + height
    x2 = x1 + width
    z2 = z1 + depth
    return np.stack([y1, x1, z1, y2, x2, z2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, z1, y2, x2, z2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 3] - box[:, 0]
    width = box[:, 4] - box[:, 1]
    depth  =box[:, 5] - box[:, 2]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width
    center_z = box[:, 2] + 0.5 * depth

    gt_height = gt_box[:, 3] - gt_box[:, 0]
    gt_width = gt_box[:, 4] - gt_box[:, 1]
    gt_depth = gt_box[:, 5] - gt_box[:, 2]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width
    gt_genter_z = gt_box[:, 2] + 0.5 * gt_depth

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dz = (gt_genter_z - center_z) / depth
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)
    dd = tf.log(gt_depth / depth)

    result = tf.stack([dy, dx,dz, dh, dw, dd], axis=1)
    return result

############################################################
#  Anchors
############################################################
def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride, num_anchors_per_location):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [depth, height, width] spatial shape of the feature map over which
            to generate anchors. The shape (BACKBONE_SHAPES) tells the resolution of the layer
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    #the cube anchor generation will work ok only for this case the ratios is [0.5, 1, 2]
    # Enumerate heights and widths from scales and ratios

    #TODO depts and should attach the logic behind the cubes someshow
    #TODO for example you should draw to visualize the following case:
    #TODO scale = 8, ratios = [0.5, 1, 2]
    #TODO heights = [ 11  8  5   5]
    #TODO weights = [  5  8 11   5]
    #TODO depths =  [  5  8  5  11]
    #basically there will be not 3 boxes but 4 cubes
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    heights = heights.tolist()
    widths = widths.tolist()
    heights.append(widths[0])
    widths.append(widths[0])
    depths = [widths[0], widths[1], widths[0], heights[0]]

    if num_anchors_per_location ==12:
        for i in range(4):
            heights.append(heights[i]*1.25)
            heights.append(heights[i]*0.75)
            widths.append(widths[i]*1.25)
            widths.append(widths[i]*0.75)
            depths.append(depths[i]*1.25)
            depths.append(depths[i]*0.75)

    heights = np.array(heights)
    widths = np.array(widths)
    depths = np.array(depths)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride + heights[0] /2
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride + widths[0] /2
    shifts_z = np.arange(0, shape[2], anchor_stride) * feature_stride + depths[0] /2
    shifts_x, shifts_y, shifts_z = np.meshgrid(shifts_x, shifts_y, shifts_z)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
    box_depths, box_centers_z = np.meshgrid(depths, shifts_z)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x, box_centers_z], axis=2).reshape([-1, 3])
    box_sizes = np.stack(
        [box_heights, box_widths, box_depths], axis=2).reshape([-1, 3])

    # Convert to corner coordinates (y1, x1, z1, y2, x2, z2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)


    # if(scales[1]==32):
    #     plotBoxesMesh(boxes)
    return boxes


def plotBoxes(boxes):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import product, combinations
    print boxes.max()
    for y1,x1,z1,y2,x2,z2 in boxes[:,:]:
        w = x2- x1
        h = y2 - y1
        d = z2 - z1
        x = x1 + w /2
        y = y1 + h/2
        z = z1 + d/2

        x, y, z = np.indices((8, 8, 8))
        tx = (x < 3)
        ty = (y < 3)
        tz = (z < 3)
        x, y, z = np.indices((128, 128, 128))
        tx = ((x1 < x) &(x < x2) )
        ty = ((y1 < y) &(y < y2))
        tz = ((z1 < z) &(z < z2))
        cube1 = tx & ty & tz
        voxels = cube1
        colors = np.empty(voxels.shape, dtype=object)
        colors[cube1] = 'red'
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor='k')
        plt.show()

def plotBoxesMesh(boxes):
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for y1,x1,z1,y2,x2,z2 in boxes[:,:]:
        points = np.array([[y1,x1,z1],
                           [y1,x2,z1],
                           [y1,x2, z2],
                           [y1, x1,z2],
                           [y2,x1, z1],
                           [y2,x2,z1],
                           [y2,x2,z2],
                           [y2,x1,z2]])

        r = [0,128]
        X, Y = np.meshgrid(r, r)
        # plot vertices
        ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
        ax.set_xlim(0,128)
        ax.set_ylim(0,128)
        ax.set_zlim(0,128)

        Z = points
        verts = [[Z[0],Z[1],Z[2],Z[3]],
                 [Z[4],Z[5],Z[6],Z[7]],
                 [Z[0],Z[1],Z[5],Z[4]],
                 [Z[2],Z[3],Z[7],Z[6]],
                 [Z[1],Z[2],Z[6],Z[5]],
                 [Z[4],Z[7],Z[3],Z[0]],
                 [Z[2],Z[3],Z[7],Z[6]]]
        ax.add_collection3d(Poly3DCollection(verts,
                                             facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
    print 'finished showing'

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride, num_anchors_per_location):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (y1, x1, z1, y2, x2, z2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, z1, y2, x2, z2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride, num_anchors_per_location))
    return np.concatenate(anchors, axis=0)


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result
