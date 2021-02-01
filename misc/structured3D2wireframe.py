import json
import scipy.io as sio
import os
import os.path as osp
import cv2
import matplotlib
matplotlib.use('Cairo')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
import argparse
import csv
import sys
import shutil
import shapely.geometry as sg
OUTPATH = '/home/davidg/epnet/data/york'
SRC = '/home/nxue2/lcnn/data/york_lcnn/valid'
EPS = 1e-10
PLANE_SEMANTIC_CLASSES = [
    'door',
    'window',
    'outwall',
    'living room',
    'kitchen',
    'bedroom',
    'bathroom',
    'balcony',
    'corridor',
    'dining room',
    'study',
    'studio',
    'store room',
    'garden',
    'laundry room',
    'office',
    'basement',
    'garage',
    'undefined']


def link_and_annotate(root, scene_dir, out_image_dir):
    with open(osp.join(root, scene_dir, 'annotation_3d.json')) as f:
        ann_3D = json.load(f)

    # Prepare ann_3D with information we want later
    ann_3D['lineJunctionMatrix'] = np.array(ann_3D['lineJunctionMatrix'], dtype=np.bool)
    ann_3D['planeLineMatrix'] = np.array(ann_3D['planeLineMatrix'], dtype=np.bool)
    ann_3D['planeJunctionMatrix'] = np.zeros([len(ann_3D['planes']), len(ann_3D['junctions'])], dtype=np.bool)
    ann_3D['junctionCoords'] = np.array([j['coordinate'] for j in ann_3D['junctions']], dtype=np.float32).T
    for plane_idx, line_mask in enumerate(ann_3D['planeLineMatrix']):
        jmask = np.any(ann_3D['lineJunctionMatrix'][line_mask], axis=0)
        ann_3D['planeJunctionMatrix'][plane_idx] = jmask

    plane_params = []
    for p in ann_3D['planes']:
        plane_params.append(p['normal'] + [p['offset']])
    ann_3D['planeParams'] = np.array(plane_params, dtype=np.float32).T
    scene_id = scene_dir.split('_')[1]


    #TODO: Remove Sanity check when done
    assert np.all([idx == plane['ID'] for (idx,plane) in enumerate(ann_3D['planes'])])
    assert np.all([idx == line['ID'] for (idx,line) in enumerate(ann_3D['lines'])])
    assert np.all([idx == junction['ID'] for (idx,junction) in enumerate(ann_3D['junctions'])])

    #Map semantics to planes and vice versa
    semantic2planeID = {s:[] for s in PLANE_SEMANTIC_CLASSES}
    for semantic in ann_3D['semantics']:
        semantic2planeID[semantic['type']].append(semantic['planeID'])
        for id in semantic['planeID']:
            assert 'semantic' not in ann_3D['planes'][id]
            ann_3D['planes'][id]['semantic'] = semantic['type']

    for type, planeIDs in semantic2planeID.items():
        semantic2planeID[type] = np.unique(planeIDs)

    plt.figure()
    fig, ax = plt.subplots(2,1)
    ax[0].set_title('XY')
    ax[1].set_title('YZ')
    filtered_line_idx = []
    for l_id, line2junc in enumerate(ann_3D['lineJunctionMatrix']):
        plane_ids = np.flatnonzero(ann_3D['planeLineMatrix'][:,l_id])
        if np.any(np.isin(plane_ids, semantic2planeID['outwall'], assume_unique=True)):
            continue
        # if ~np.any(np.isin(plane_ids, semantic2planeID['window'], assume_unique=True)):
        #     continue
        filtered_line_idx.append(l_id)
        idx1, idx2 = np.flatnonzero(line2junc)
        j1 = ann_3D['junctions'][idx1]['coordinate']
        j2 = ann_3D['junctions'][idx2]['coordinate']
        ax[0].plot((j1[0], j2[0]), (j1[1], j2[1]))
        ax[1].plot((j1[1], j2[1]), (j1[2], j2[2]))
    plot_scene_dir = '/host_home/plots/hawp/{}'.format(scene_id)
    os.makedirs(plot_scene_dir, exist_ok=True)
    plt.savefig(osp.join(plot_scene_dir, 'scene.png'))


    ann_3D['semantic2planeID'] = semantic2planeID
    ann_3D['filtered_line_idx'] = filtered_line_idx
    out_ann = []

    render_dir = osp.join(root, scene_dir, '2D_rendering')
    for room_id in os.listdir(render_dir):
        room_dir = osp.join(render_dir, room_id, 'perspective', 'full')
        for pos_id in os.listdir(room_dir):
            ann = {}
            pos_dir = osp.join(room_dir, pos_id)
            img_name = 'S{}R{:0>5s}P{}.png'.format(scene_id, room_id, pos_id)
            img = Image.open(osp.join(pos_dir,'rgb_rawlight.png'))
            ann['filename'] = img_name
            ann['width'], ann['height'] = img.size
            os.symlink(
                osp.join(pos_dir,'rgb_rawlight.png'),
                osp.join(out_image_dir, img_name))

            with open(osp.join(pos_dir, 'layout.json')) as f:
                ann_2D = json.load(f)

            with open(osp.join(pos_dir, 'camera_pose.txt')) as f:
                pose = next(csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC))

            plt.figure()
            plt.imshow(img)
            add_line_annotation(ann_3D, ann_2D, pose, ann)
            # plt.legend()
            plt.savefig(osp.join(plot_scene_dir, 'R{}P{}_3D_proj.png'.format(room_id, pos_id)))

            plt.figure()
            plt.imshow(img)
            plt.plot(ann['junc'][:,0], ann['junc'][:,1], '.')
            for edge in ann['edges_positive']:
                plt.plot((ann['junc'][edge[0],0], ann['junc'][edge[1],0]),
                         (ann['junc'][edge[0],1], ann['junc'][edge[1],1]))

            plt.savefig(osp.join(plot_scene_dir, 'R{}P{}_2D.png'.format(room_id, pos_id)))
            out_ann.append(ann)

    return out_ann

def add_line_annotation(ann_3D, ann_2D, pose, out):
    R, t, K = parse_camera_info(pose, out['width'], out['height'])
    K_inv = np.linalg.inv(K)
    T = np.block([
        [R, t],
        [np.array([0,0,0,1])]
    ])
    T_inv = np.linalg.inv(T)

    # wall_mask = np.array([p['type'] == 'wall' for p in ann_3D['planes']], dtype=np.bool)
    # wall_planes = T@ann_3D['planeParams'][:,wall_mask]
    img_planes = T_inv.T@ann_3D['planeParams']
    img_junctions = R@ann_3D['junctionCoords'] + t
    plane_junctions = [img_junctions[:,mask] for mask in ann_3D['planeJunctionMatrix']]
    img_poly = sg.box(0,0,out['width'], out['height'])
    junctions = []
    edges_pos = []
    # Intersect each line with image
    for l_idx in ann_3D['filtered_line_idx']:
        # Find junctions for line
        j12_img = img_junctions[:,ann_3D['lineJunctionMatrix'][l_idx]]

        behind, j12_img = line_to_front(j12_img)

        # Check if line was in front of camera
        if j12_img is None:
            continue

        print('first')
        modified, j12_img_px = line_to_img(j12_img, K, img_poly = img_poly)

        # Check if line was inside image bounds
        if j12_img_px is None:
            continue

        # print(j12_img)
        # j12_img = K_inv@np.vstack([j12_img_px, np.ones([1,2])])
        # print(j12_img)

        #Check occluding planes for overlap in image
        occluded, j12_img_visible = is_line_behind_planes(j12_img, img_planes, plane_junctions)
        if j12_img_visible is None:
            continue

        # j12_img_px = K@j12_img_visible
        # j12_img_px = j12_img_px[:2] / j12_img_px[2]
        print('second')
        modified, j12_img_px = line_to_img(j12_img_visible, K, img_poly = img_poly)

        #Check if line was inside image bounds
        if j12_img_px is None:
            continue


        # occluded = np.zeros_like(modified)
        true_junction = ~(modified | behind | occluded)
        plt.plot(*j12_img_px, linestyle='solid', color='b', label='Line')
        for idx in range(2):
            if true_junction[idx]:
                plt.plot(*j12_img_px[:,idx], marker='o', color='b', label='true junction')
            else:
                if behind[idx]:
                    plt.plot(*j12_img_px[:,idx], marker='o', color='r', label='behind')
                if modified[idx]:
                    plt.plot(*j12_img_px[:,idx], marker='x', color='k', label='modified')
                if occluded[idx]:
                    plt.plot(*j12_img_px[:,idx], marker='+', color='c', label='occluded')

    out['junc'] = img_junctions
    out['edges_positive'] = edges_pos

def line_to_front(line_points):
    """ Project a line segment in 3D to be in front of camera, assuming line is in camera homegenous coordinates.
    I.e. Camera position is at origin.
    line_points: 3x2, each column being a point.
    """
    behind = line_points[2] < 0

    # Both end-points behind camera?
    if np.all(line_points[2] < 0):
        return behind, None

    line_points = np.copy(line_points)

    if line_points[2,0] < 0:
        q = line_points[:,0] - line_points[:,1]
        line_points[:,0] = line_points[:,1] -(line_points[2,1]/(q[2]+EPS))*q
    elif line_points[2,1] < 0:
        q = line_points[:,1] - line_points[:,0]
        line_points[:,1] = line_points[:,0] -(line_points[2,0]/(q[2]+EPS))*q

    return behind, line_points

def line_to_img(line_points, K, img_poly = None, width = None, height = None):
    """ Project a line segment in 3D FOV in image pixels.  Assumes camera is at origin and line in front of camera.
    line_points: 3x2, each column being a point.
    """
    if not np.all(line_points[2] > -EPS):
        print(line_points[2])
    assert np.all(line_points[2] > -EPS)

    # Construct image polygon if not supplied
    if not img_poly:
        img_poly = sg.box(0,0,width, height)

    # Line in Pixel coordinates
    line_points_px = K@line_points
    line_points_px /= (line_points_px[2] + EPS)
    line = sg.LineString(line_points_px[:2].T)

    # Line in image bounds
    line_img = img_poly.intersection(line)

    # Check if line was inside image bounds
    modified = np.zeros(2, dtype=np.bool)
    if line_img.is_empty:
        new_line_points = None
    else:
        new_line_points = np.array(line_img.coords).T
        for i in range(2):
            modified[i] |= not line_img.boundary[i].equals(line.boundary[i])

    return modified, new_line_points


def is_line_behind_planes(line_points, planes, plane_junction_list):
    """ Assumes camera is at origin, checks if line is visible due to planes.
    Assumes line in front of camera.
    line_points: 3x2, each column being a point.
    planes: 4xN coefficients for N planes.
    plane_junction_list: List with 3xM junctions in each element, M may vary between planes
    """
    assert np.all(line_points[2] > -EPS)

    endp_occluded = np.zeros(2, dtype=np.bool)
    new_line_points = line_points.copy()

    # TODO: Note that each line may be split into multiple lines
    for p_idx, plane_junctions in enumerate(plane_junction_list):
        plane = planes[:,p_idx]

        # 0 < dist_frac < 1 => plane between camera and point
        dist_frac = - plane[3]/(plane[:3]@line_points + EPS)
        occluded = (0 < dist_frac) & (dist_frac < 1)
        # print('Frac:', dist_frac, 'Occluded: ', occluded)
        if not np.any(occluded):
            continue

        endp_plane = dist_frac*line_points
        if ~np.all(occluded):

            # If one of the points are in front of the plane we need to find the intersection
            # between line and plane
            endp_plane2 = endp_plane.copy()
            endp_idx = np.flatnonzero(~occluded)
            print(occluded, endp_idx)
            line_v = line_points[:,1,None]-line_points[:,0,None]
            line_dist = -(plane[:3]@line_points[:,0] + plane[3])/(plane[:3]@line_v)
            endp_plane2[:,endp_idx] = line_points[:,0,None] + line_dist*line_v
            plane_eq = np.abs(plane[:3]@(line_points[:,0,None] + line_dist*line_v) + plane[3])
            if plane_eq > EPS:
                print(plane_eq)
                print(line_dist)
            assert np.linalg.norm(plane[:3]@plane_junctions + plane[3]) < 1e-5
            # assert plane_eq < EPS

            fig = plt.figure()
            ax = fig.add_subplot(221, projection='3d')
            ax.set_title('dist_frac {}, line dist {}'.format(dist_frac, line_dist))
            ax.plot(0,0,0,'ko')
            ax.plot(*line_points, 'b.-')
            ax.plot(*endp_plane, 'g.-')
            ax.plot(*endp_plane2, 'gx-')
            ax.plot(*plane_junctions, 'ro')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax = fig.add_subplot(222)
            ax.set_title('XY')
            ax.plot(*line_points[:2], 'b.-')
            ax.plot(*endp_plane[:2], 'g.-')
            ax.plot(*endp_plane2[:2], 'gx-')
            ax.plot(*plane_junctions[:2], 'ro')
            ax.plot(0,0,'ko')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax = fig.add_subplot(223)
            ax.set_title('YZ')
            ax.plot(*line_points[1:], 'b.-')
            ax.plot(*endp_plane[1:], 'g.-')
            ax.plot(*endp_plane2[1:], 'gx-')
            ax.plot(*plane_junctions[1:], 'ro')
            ax.plot(0,0,'ko')
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax = fig.add_subplot(224)
            ax.set_title('ZX')
            ax.plot(*line_points[[2,0]], 'b.-')
            ax.plot(*endp_plane[[2,0]], 'g.-')
            ax.plot(*endp_plane2[[2,0]], 'gx-')
            ax.plot(*plane_junctions[[2,0]], 'ro')
            ax.plot(0,0,'ko')
            ax.set_xlabel('Z')
            ax.set_ylabel('X')
            plt.tight_layout()


            plt.savefig('/host_home/plots/hawp/debug/3D_{:03d}.svg'.format(p_idx))
            plt.close(fig)
            print('dist_frac', dist_frac)
            print('line_dist', line_dist)
            assert ((0 <= line_dist) and (line_dist <= 1))





        #Only larger dist_frac for development
        # print('Found plane occluding')
        # print(dist_frac)

        # Project point on plane and take intersection
        # Align X axis with line and Z axis with plane normal

        # print(plane.shape, endp_plane.shape)
        # print('Endpoints on plane:', plane[:3]@endp_plane + plane[3])
        # print('Junctions on plane:',plane[:3]@plane_junctions + plane[3])
        #Move origo to first end point
        # t = np.zeros_like(t)
        W = normalize(plane[:3])
        U = normalize(endp_plane[:,0] - endp_plane[:,1])
        # print(W.shape, U.shape)
        V = np.cross(W,U)
        R = np.vstack([U,V,W])
        t = -R@endp_plane[:,0].reshape([3,1])
        T = np.block([
            [R, t],
            [np.array([0,0,0,1])]
        ])
        T_inv = np.linalg.inv(T)
        # T_inv = T.T
        plane2 = T_inv.T@plane
        in_plane_junctions = R@plane_junctions + t
        in_plane_endp = R@endp_plane + t

        # inv_in_plane_endp = R.T@(in_plane_endp - t)
        # if ~np.all(np.abs(inv_in_plane_endp - endp_plane) < 1e-5):
        #     print(np.abs(inv_in_plane_endp - endp_plane))
        #
        # assert np.all(np.abs(inv_in_plane_endp - endp_plane) < 1e-5)

        # print(plane_junctions.shape)
        # print(in_plane_junctions[0])
        # print(in_plane_junctions[1])
        # print(in_plane_junctions[2])
        # print(in_plane_endp)
        # print('Endpoints on plane:', plane2[:3]@in_plane_endp + plane2[3])
        # print('Junctions on plane:',plane2[:3]@in_plane_junctions + plane2[3])

        # fig = plt.figure()
        p_poly_sg = sg.Polygon(in_plane_junctions[:2].T).convex_hull
        p_line_sg = sg.LineString(in_plane_endp[:2].T)
        # plt.plot(*np.array(p_poly_sg.boundary.coords).T, 'b.-')
        # plt.plot(*np.array(p_line_sg.coords).T, 'r.-')

        if p_line_sg.disjoint(p_poly_sg):
            # plt.savefig('/host_home/plots/hawp/debug/disjoint{:03d}.svg'.format(p_idx))
            # plt.close(fig)
            continue

        line_segment = p_line_sg.difference(p_poly_sg)
        if not isinstance(line_segment, sg.LineString):
            #Cannot handle edge case yet
            # for ls in line_segment:
            #     plt.plot(*np.array(ls.coords).T, 'g.-')
            # plt.savefig('/host_home/plots/hawp/debug/multisegment{:03d}.svg'.format(p_idx))
            # plt.close(fig)
            print(type(line_segment))
            new_line_points = None
            break

        if line_segment.is_empty:
            new_line_points = None
            # plt.savefig('/host_home/plots/hawp/debug/empty{:03d}.svg'.format(p_idx))
            # plt.close(fig)
            break #exit here, no line left
        else:
            # zval = in_plane_endp[2,0]
            # plt.plot(*np.array(line_segment.coords).T, 'g.-')
            zval = 0
            in_plane_seg = np.vstack([np.array(line_segment.coords).T, zval*np.ones([1,2])])
            new_line_points = R.T@(in_plane_seg - t)
            if np.any(new_line_points[2] < 0 ):
                print("Error")
                print('Dist frac')
                print(dist_frac)
                print("Line prior projeciton")
                print(line_points)
                print('Line ')
                print(endp_plane)
                print('Line in plane ')
                print(in_plane_endp)
                print('cut Line in plane')
                print(in_plane_seg)
                print('cut Line in world')
                print(new_line_points)
                # plt.savefig('/host_home/plots/hawp/debug/negative{:03d}.svg'.format(p_idx))
                # plt.close(fig)
                sys.exit()
            for i in range(2):
                endp_occluded[i] |= not line_segment.boundary[i].equals(p_line_sg.boundary[i])

        # plt.savefig('/host_home/plots/hawp/debug/adjusted{:03d}.svg'.format(p_idx))
        # plt.close(fig)

    return endp_occluded, new_line_points



        # p_junc /= p_junc[2] + eps
        # p_poly = sg.Polygon(p_junc[:2].T).convex_hull
        # p_poly = p_poly.intersection(img_poly)
        #
        # plt.plot(*p_poly.exterior.xy, linestyle='dashed')



def pflat(points):
    return points/points[-1]

def pextend(points):
    return np.vstack([points, np.ones([1,points.shape[1]])])

def normalize(vector):
    return vector / np.linalg.norm(vector)

def parse_camera_info(camera_info, width, height):
    """ extract intrinsic and extrinsic matrix
    Make K, R and t s.t. lambda*x = K*(R*X + t)
    where lambda is any scalar, x is point in image in pixels and X is point in world coordinates
    """
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = np.cross(W, up)
    V = np.cross(W, U)

    R = np.vstack((U, V, W))
    # print(R@U.T)
    # print(R@V.T)
    # print(R@up.T)
    # print(R@W.T)

    camera_pos = np.array(camera_info[:3]).reshape(3,1)

    t = -R@camera_pos

    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    return R, t, K

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate wireframe format from Structured3D')
    parser.add_argument('data_dir', type=str, help='Path to Structured3D')
    parser.add_argument('out_dir', type=str, help='Path to storing conversion')
    parser.add_argument('-j', '--nbr-workers', type=int, default = 1, help='Number of processes to split work on')
    parser.add_argument('-s', '--nbr-scenes', type=int, default = None, help='Number of scenes to process')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite out_dir if existing')

    args = parser.parse_args()

    if osp.exists(args.out_dir) and not args.overwrite:
        print("Output directory {} already exists, specify -o flag if overwrite is permitted".format(args.out_dir))
        sys.exit()

    shutil.rmtree(args.out_dir)
    out_image_dir = osp.join(args.out_dir, 'images')
    os.makedirs(out_image_dir)

    dirs = sorted(os.listdir(args.data_dir))
    if args.nbr_scenes:
        dirs = dirs[:args.nbr_scenes]

    for scene_dir in dirs:
        link_and_annotate(args.data_dir, scene_dir, out_image_dir)
