from nifti import *
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib
import matplotlib.pyplot as plt

def draw(data, y1, x1, z1, y2, x2, z2, colour, text):

    front_side = data[:,:,z1]
    back_side = data[:,:,z2]
    front_side = np.array(front_side,dtype=np.float32)/ data.max()
    back_side = np.array(back_side,dtype=np.float32)/ data.max()
    front_side = cv2.rectangle(front_side,(x1,y1),(x2,y2),colour,2)
    back_side = cv2.rectangle(back_side,(x1,y1),(x2,y2),colour,2)
    cv2.imshow(text+" front side",front_side)
    cv2.imshow(text+" back side", back_side)
    cv2.waitKey(100)

    left_side = data[:,x1,:]
    right_side = data[:,x2,:]
    left_side = np.array(left_side,dtype=np.float32)/ data.max()
    right_side = np.array(right_side,dtype=np.float32)/ data.max()
    left_side = cv2.rectangle(left_side,(z1,y1),(z2,y2),colour,2)
    right_side = cv2.rectangle(right_side,(z1,y1),(z2,y2),colour,2)
    cv2.imshow(text+" left_side",left_side)
    cv2.imshow(text+" right_side", right_side)
    cv2.waitKey(100)

    top_side = data[y1,:,:]
    down_side = data[y2,:,:]
    top_side=  np.array(top_side,dtype=np.float32)/ data.max()
    down_side = np.array(down_side,dtype=np.float32) / data.max()
    top_side = cv2.rectangle(top_side, (z1,x1),(z2,x2),colour,2)
    down_side = cv2.rectangle(down_side, (z1,x1),(z2,x2),colour,2)
    cv2.imshow(text+" top_side",top_side)
    cv2.imshow(text+" down_side", down_side)
    cv2.waitKey(100)


def validateNNOutput(data, out_bbox, gt_bbox,sleep_time=100000):
    y1 = int(out_bbox[0,0])
    x1 = int(out_bbox[0,1])
    z1 = int(out_bbox[0,2])
    y2 = int(out_bbox[0,3])
    x2 = int(out_bbox[0,4])
    z2 = int(out_bbox[0,5])
    draw(data, y1, x1, z1, y2, x2, z2, (255), "NN ")
    y1 = int(gt_bbox[0,0])
    x1 = int(gt_bbox[0,1])
    z1 = int(gt_bbox[0,2])
    y2 = int(gt_bbox[0,3])
    x2 = int(gt_bbox[0,4])
    z2 = int(gt_bbox[0,5])
    draw(data, y1, x1, z1, y2, x2, z2, (255), "GT ")
    cv2.waitKey(sleep_time)


def visualize_target_graphs(rois, gt_boxes):

    for i in reversed(range(len(rois))):
        y1,x1,z1,y2,x2,z2 = gt_boxes[i,:6]
        Y1,X1,Z1,Y2,X2,Z2 = rois[i,:6]
        dist = 30
        volume_roi = (Y2-Y1)*(X2-X1) * (Z2 -Z1)
        if not( abs(Y1-y1) <dist and abs(X1-x1) <dist and abs(Z1-z1) < dist and abs(Y2-y2) < dist and abs(X2-x2) < dist  and abs(Z2 - z2) < dist and max_number>0 and volume_roi > 15*15*15):
            continue
        max_number= max_number-1
        points = np.array([[y1,x1,z1],
                           [y1,x2,z1],
                           [y1,x2, z2],
                           [y1, x1,z2],
                           [y2,x1, z1],
                           [y2,x2,z1],
                           [y2,x2,z2],
                           [y2,x1,z2]])

        points_R = np.array([[Y1,X1,Z1],
                             [Y1,X2,Z1],
                             [Y1,X2,Z2],
                             [Y1,X1,Z2],
                             [Y2,X1,Z1],
                             [Y2,X2,Z1],
                             [Y2,X2,Z2],
                             [Y2,X1,Z2]])


        # plot vertices
        ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
        ax.scatter3D(points_R[:, 0], points_R[:, 1], points_R[:, 2])
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
        Z = points_R
        verts = [[Z[0],Z[1],Z[2],Z[3]],
                 [Z[4],Z[5],Z[6],Z[7]],
                 [Z[0],Z[1],Z[5],Z[4]],
                 [Z[2],Z[3],Z[7],Z[6]],
                 [Z[1],Z[2],Z[6],Z[5]],
                 [Z[4],Z[7],Z[3],Z[0]],
                 [Z[2],Z[3],Z[7],Z[6]]]
        ax.add_collection3d(Poly3DCollection(verts,
                                             facecolors='red', linewidths=1, edgecolors='r', alpha=.25))


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        print 'finished showing'
    plt.show()
    time.sleep(1000)
    return rois_orig, gt_boxes_orig

def showBoxes(rois,ax, i, colour,edgecolors):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    Y1,X1,Z1,Y2,X2,Z2 = rois[i,:6]
    points_R = np.array([[Y1,X1,Z1],
                         [Y1,X2,Z1],
                         [Y1,X2,Z2],
                         [Y1,X1,Z2],
                         [Y2,X1,Z1],
                         [Y2,X2,Z1],
                         [Y2,X2,Z2],
                         [Y2,X1,Z2]])
    ax.scatter3D(points_R[:, 0], points_R[:, 1], points_R[:, 2])
    ax.set_xlim(0,128)
    ax.set_ylim(0,128)
    ax.set_zlim(0,128)
    Z = points_R
    verts = [[Z[0],Z[1],Z[2],Z[3]],
             [Z[4],Z[5],Z[6],Z[7]],
             [Z[0],Z[1],Z[5],Z[4]],
             [Z[2],Z[3],Z[7],Z[6]],
             [Z[1],Z[2],Z[6],Z[5]],
             [Z[4],Z[7],Z[3],Z[0]],
             [Z[2],Z[3],Z[7],Z[6]]]
    ax.add_collection3d(Poly3DCollection(verts,
                                         facecolors=colour, linewidths=1, edgecolors=edgecolors, alpha=.25))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


trans = 0.5
class_colors = {1:(0.0,0.8,0.0,trans), 2:(1.0,0.0,1.0,trans), 3:(1.0,1.0,0.0,trans), 4:(0.0,0.0,0.0,trans), 5:(1.0,1.0,1.0,trans), 6:(0.0,0.0,1.0,trans)}
def visualizeNNOutput(out_bbox, class_ids, gt_bbox,sleep_time=100000):

    patch_lung1 = mpatches.Patch(color=(0.0,0.8,0.0), label='Left lung')
    patch_lung2 = mpatches.Patch(color=(1.0,0.0,1.0), label='right lung')
    patch_liver = mpatches.Patch(color=(1.0,1.0,0.0), label='Liver')
    patch_kidney1 = mpatches.Patch(color=(0.0,0.0,0.0), label='Left kidney')
    patch_kidney2 = mpatches.Patch(color=(1.0,1.0,1.0), label='Right kidney')
    patch_heart = mpatches.Patch(color=(0.0,0.0,1.0), label='Heart')

    # fig = plt.figure()
    # fig.legend(handles=[patch_lung1, patch_lung2, patch_liver, patch_kidney1, patch_kidney2,patch_heart])
    # ax = fig.add_subplot(111, projection='3d')
    # rois =out_bbox
    # print len(rois)
    # for i in range(len(rois)):
    #     showBoxes(rois,ax, i, colour=class_colors[class_ids[i]] , edgecolors="cyan")
    # fig2 = plt.figure()
    # fig2.legend(handles=[patch_lung1, patch_lung2, patch_liver, patch_kidney1, patch_kidney2,patch_heart])
    # ax2 = fig2.add_subplot(111, projection='3d')
    # for i in range(len(gt_bbox)):
    #     class_id = gt_bbox[i,6]
    #     showBoxes(gt_bbox,ax2,i,colour=class_colors[class_id], edgecolors="red")
    # plt.show()
    fig = plt.figure(num=None, figsize=(8, 16), dpi=80, facecolor='w', edgecolor='k')
    move_figure(fig, 500, 500)
    fig.legend(handles=[patch_lung1, patch_lung2, patch_liver, patch_kidney1, patch_kidney2,patch_heart])
    ax = fig.add_subplot(211, projection='3d')
    rois =out_bbox
    print len(rois)
    for i in range(len(rois)):
        showBoxes(rois,ax, i, colour=class_colors[class_ids[i]] , edgecolors="red")


    ax2 = fig.add_subplot(212, projection='3d')
    for i in range(len(gt_bbox)):
        class_id = gt_bbox[i,6]
        showBoxes(gt_bbox,ax2,i,colour=class_colors[class_id], edgecolors="red")
    plt.show()