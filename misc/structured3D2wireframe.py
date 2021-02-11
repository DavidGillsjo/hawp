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
import shapely
OUTPATH = '/home/davidg/epnet/data/york'
SRC = '/home/nxue2/lcnn/data/york_lcnn/valid'
CMP_EPS = 1e-5
DIV_EPS = 1e-15
P_IDX = 0
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
            plt.colorbar()
            plt.savefig(osp.join(plot_scene_dir, 'R{}P{}_3D_proj.png'.format(room_id, pos_id)))


            plt.figure()
            plt.imshow(img)
            coords2d = np.array([j['coordinate'] for j in ann_2D['junctions']])
            plt.plot(*coords2d.T, '.')
            # plt.plot(ann['junc'][:,0], ann['junc'][:,1], '.')
            # for edge in ann['edges_positive']:
            #     plt.plot((ann['junc'][edge[0],0], ann['junc'][edge[1],0]),
            #              (ann['junc'][edge[0],1], ann['junc'][edge[1],1]))
            #
            plt.savefig(osp.join(plot_scene_dir, 'R{}P{}_2D.png'.format(room_id, pos_id)))
            out_ann.append(ann)
            # return out_ann #DEBUG

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

        modified, j12_img = line_to_front(j12_img)
        planes_mask = ann_3D['planeLineMatrix'][:,l_idx]

        # Check if line was in front of camera
        if j12_img is None:
            continue

        _, j12_img_px = line_to_img(j12_img, K, img_poly = img_poly)

        # Check if line was inside image bounds
        if j12_img_px is None:
            continue

        # print(j12_img)
        # j12_img = K_inv@np.vstack([j12_img_px, np.ones([1,2])])
        # print(j12_img)
        #Check occluding planes for overlap in image
        modified_list, segment_list = get_visible_segments(modified, j12_img, img_planes, plane_junctions, planes_mask, l_idx = l_idx)
        if not segment_list:
            continue

        print('------------------ l_idx {} ------------------'.format(l_idx))
        print('Line semantics', [ann_3D['planes'][id]['semantic'] for id in np.flatnonzero(planes_mask)])
        print('Modified line to front', modified)

        # j12_img_px = K@j12_img_visible
        # j12_img_px = j12_img_px[:2] / j12_img_px[2]
        modified_list_pix = []
        segment_list_pix = []
        segment_list_z = []
        for mod, seg in zip(modified_list, segment_list):
            print('Modified occlusion', mod)
            mod_pix, seg_pix = line_to_img(seg, K, img_poly = img_poly, l_idx=l_idx)
            print('Modified to img', mod_pix)
            if seg_pix is not None:
                segment_list_pix.append(seg_pix)
                modified_list_pix.append(mod | mod_pix)
                segment_list_z.append(seg[2])


        #Check if line was inside image bounds
        if not segment_list_pix:
            continue

        for mod, seg, z in zip(modified_list_pix, segment_list_pix, segment_list_z):
            print('Z: ', z)
            # occluded = np.zeros_like(modified)
            # true_junction = ~(modified | behind | occluded)
            plt.plot(*seg, linestyle='solid', color='b', label='Line')
            plt.text(*np.mean(seg, axis=1), str(l_idx), rotation=45)
            # plt.scatter(*seg, c=z)
            for idx in range(2):
                if mod[idx]:
                    plt.plot(*seg[:,idx], marker='.', color='r', label='behind')
                else:
                    plt.plot(*seg[:,idx], marker='.', color='b', label='true junction')


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
        line_points[:,0] = line_points[:,1] -(line_points[2,1]/(q[2]+DIV_EPS))*q
        line_points[2,0] = 0 # Make sure it is actually 0 since this assumed in occlusion check
    elif line_points[2,1] < 0:
        q = line_points[:,1] - line_points[:,0]
        line_points[:,1] = line_points[:,0] -(line_points[2,0]/(q[2]+DIV_EPS))*q
        line_points[2,1] = 0 # Make sure it is actually 0 since this assumed in occlusion check

    return behind, line_points

def line_to_img(line_points, K, img_poly = None, width = None, height = None, l_idx = None):
    """ Project a line segment in 3D FOV in image pixels.  Assumes camera is at origin and line in front of camera.
    line_points: 3x2, each column being a point.
    """
    assert np.all(line_points[2] > -CMP_EPS)

    # Construct image polygon if not supplied
    if not img_poly:
        img_poly = sg.box(0, 0, width, height)

    # Line in Pixel coordinates
    line_points_px = K@line_points
    line_points_px /= (line_points_px[2] + DIV_EPS)
    line = sg.LineString(line_points_px[:2].T)

    # Line in image bounds
    line_img = img_poly.intersection(line)

    # Check if line was inside image bounds
    unmodified = np.zeros(2, dtype=np.bool)
    if line_img.is_empty or isinstance(line_img, sg.Point):
        new_line_points = None
    else:
        new_line_points = np.array(line_img.coords).T
        for i in range(2):
            unmodified |= (np.linalg.norm(line_points_px[:2] - new_line_points[:,i,None], axis=0) < 1e-5)
        # if l_idx:
        #     plt.figure()
        #     plt.plot(*np.array(img_poly.boundary.coords).T, 'b.-')
        #     plt.plot(*line_points_px[:2], 'r.-')
        #     plt.plot(*new_line_points, 'g.-')
        #     for i in range(2):
        #             # plt.text(*new_line_points[:,i], 'Diff: {}'.format(line_points_px[:2] - new_line_points[:,i]))
        #             plt.text(*new_line_points[:,i], 'Diff: {}'.format(np.linalg.norm(line_points_px[:2] - new_line_points[:,i,None], axis=0)), rotation=45)
        #             plt.plot(*new_line_points[:,i], 'b.' if unmodified[i] else 'b*')
        #     plt.title('Points: {}'.format(line_points_px[:2]))
        #     plt.savefig('/host_home/plots/hawp/debug/toline_{:03d}.svg'.format(l_idx))
        #     plt.close()

    return ~unmodified, new_line_points


def get_visible_segments(modified, line_points, planes, plane_junction_list, plane_line_mask, l_idx = None):
    """ Assumes camera is at origin, checks if line is visible due to planes.
    Assumes line in front of camera.
    line_points: 3x2, each column being a point.
    planes: 4xN coefficients for N planes.
    plane_junction_list: List with 3xM junctions in each element, M may vary between planes
    plane_line_mask: Bool vector of which planes the line belong to.
    """
    assert np.all(line_points[2] > -CMP_EPS)

    modified_out = [modified.copy()]
    line_segments_out = [line_points.copy()]
    P_IDX = 0

    # For each plane check if the line segment is occluded
    for p_idx, (plane, plane_junctions) in enumerate(zip(planes.T, plane_junction_list)):

        if plane_line_mask[p_idx]:
            #Skip check if line is in plane
            continue

        valid_modified = []
        valid_segments = []
        for line_points, modified in zip(line_segments_out, modified_out):
            add_visible_segments_single_plane(modified, line_points, plane, plane_junctions, valid_modified, valid_segments, P_IDX=P_IDX, l_idx = l_idx)
        modified_out = valid_modified
        line_segments_out = valid_segments
        P_IDX += 1

        # Stop if there are no valid segments left
        if not line_segments_out:
            break

    return modified_out, line_segments_out

# DBG_LIDX = 54
DBG_LIDX = -1
def add_visible_segments_single_plane(modified, line_points, plane, plane_junctions, valid_modified, valid_segments, P_IDX = 0, l_idx = None):
    # 0 < dist_frac < 1 => plane between camera and point

    assert np.all(line_points[2] > -CMP_EPS)
    #Make all z positive
    line_points[2] = np.abs(line_points[2])

    #Figure out where the viewing ray cuts the plane
    dist_frac = - plane[3]/(plane[:3]@line_points + DIV_EPS)
    occluded = (CMP_EPS < dist_frac) & (dist_frac < 1 - CMP_EPS)

    if l_idx == DBG_LIDX: print('Frac:', dist_frac, 'Occluded: ', occluded)
    # print('Frac:', dist_frac, 'Occluded: ', occluded)
    if not np.any(occluded):
        # No occlusion, skip
        valid_modified.append(modified)
        valid_segments.append(line_points)
        if l_idx == DBG_LIDX: print('OK')
        return


    # Projection to plane along viewing ray.
    endp_plane = dist_frac*line_points
    # print('line_points', line_points)
    # print('endp_plane', endp_plane)

    if ~np.all(occluded):
        """
        If one of the points are in front of the plane we need to find the intersection
        between line and plane and study the occluded segment
        """
        # endp_plane2 = endp_plane.copy()
        free_idx = np.flatnonzero(~occluded)[0]
        mod_idx = (free_idx + 1) % 2
        line_v = line_points[:,mod_idx]-line_points[:,free_idx]
        line_dist = -(plane[:3]@line_points[:,free_idx] + plane[3])/(plane[:3]@line_v + DIV_EPS)
        endp_plane[:,free_idx] = line_points[:,free_idx] + line_dist*line_v
        plane_eq = np.abs(plane[:3]@(line_points[:,free_idx] + line_dist*line_v) + plane[3])


        if line_dist < CMP_EPS:
            # No part of the line is free
            free_idx = None
            mod_idx = None
            if l_idx == DBG_LIDX: print('No part free')
        elif np.abs(line_dist - 1) < CMP_EPS:
            # All line is free
            valid_modified.append(modified)
            valid_segments.append(line_points)
            if l_idx == DBG_LIDX: print('OK')
            return

        # if not ((-CMP_EPS < line_dist) and (line_dist < 1 + CMP_EPS)):
        if l_idx == DBG_LIDX:
            fig = plt.figure()
            ax = fig.add_subplot(221, projection='3d')
            ax.set_title('dist_frac {}, line dist {}'.format(dist_frac, line_dist))
            ax.plot(0,0,0,'ko')
            ax.plot(*line_points, 'b.-')
            ax.plot(*endp_plane, 'g.-')
            ax.plot(*plane_junctions, 'ro')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax = fig.add_subplot(222)
            ax.set_title('XY')
            ax.plot(*line_points[:2], 'b.-')
            ax.plot(*endp_plane[:2], 'g.-')
            ax.plot(*plane_junctions[:2], 'ro')
            ax.plot(0,0,'ko')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax = fig.add_subplot(223)
            ax.set_title('YZ')
            ax.plot(*line_points[1:], 'b.-')
            ax.plot(*endp_plane[1:], 'g.-')
            ax.plot(*plane_junctions[1:], 'ro')
            ax.plot(0,0,'ko')
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax = fig.add_subplot(224)
            ax.set_title('ZX')
            ax.plot(*line_points[[2,0]], 'b.-')
            ax.plot(*endp_plane[[2,0]], 'g.-')
            ax.plot(*plane_junctions[[2,0]], 'ro')
            ax.plot(0,0,'ko')
            ax.set_xlabel('Z')
            ax.set_ylabel('X')
            plt.tight_layout()

            plt.savefig('/host_home/plots/hawp/debug/3D_{:03d}.svg'.format(P_IDX))
            plt.close(fig)

        # print(plane_eq)
        assert plane_eq < CMP_EPS
        # print(line_dist)
        assert ((-CMP_EPS < line_dist) and (line_dist < 1 + CMP_EPS))
        # print('line_dist', line_dist)
        if l_idx == DBG_LIDX: print('One occluded')
    else:
        free_idx = None
        mod_idx = None
        if l_idx == DBG_LIDX: print('All occluded')
        # print('All occluded')


    """
    Project point on plane and take intersection
    Align X axis with line and Z axis with plane normal
    Move origo to first end point
    """
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


    p_poly_sg = sg.Polygon(in_plane_junctions[:2].T).convex_hull
    p_line_sg = sg.LineString(in_plane_endp[:2].T)


    if p_line_sg.disjoint(p_poly_sg):
        """ No occlusion, line and plane are not overlapping """
        valid_modified.append(modified)
        valid_segments.append(line_points)
        assert np.linalg.norm(line_points[:,0] - line_points[:,1]) > 1e-5
        if l_idx == DBG_LIDX: print('No overlap')
        return


    """
    We know that there is some occlusion, we can now add the
    non-occluded section if only one endpoint was behind the plane
    """
    if free_idx is not None:
        new_modified = modified.copy()
        new_modified[mod_idx] = True
        new_line_points = line_points.copy()
        new_line_points[:,mod_idx] = endp_plane[:,free_idx]
        valid_modified.append(new_modified)
        valid_segments.append(new_line_points)
        assert np.linalg.norm(new_line_points[:,0] - new_line_points[:,1]) > 1e-5
        if l_idx == DBG_LIDX: print('Partly occluded')

        #Adjust modified to reflect the remaining segment
        modified[free_idx] = True

    visible_segments = p_line_sg.difference(p_poly_sg)

    if isinstance(visible_segments, sg.LineString):
        # There is only one segment
        visible_segments = [visible_segments]

    if l_idx == DBG_LIDX:
        fig = plt.figure()
        plt.plot(*np.array(p_poly_sg.boundary.coords).T, 'b.-')
        plt.plot(*np.array(p_line_sg.coords).T, 'r.-')


    for line_segment in visible_segments:
        if line_segment.is_empty or (line_segment.length < 1e-5):
            # No visible segment
            if l_idx == DBG_LIDX: print('No visible segment')
            continue


        # zval = in_plane_endp[2,0]

        zval = 0
        in_plane_seg = np.vstack([np.array(line_segment.coords).T, zval*np.ones([1,2])])
        new_line_points = R.T@(in_plane_seg - t)
        unmodified = np.zeros_like(modified)
        original_idx = np.flatnonzero(~modified)
        #TODO:Better loop
        for oidx in original_idx:
            original_coord = in_plane_endp[:, oidx]
            for j in range(2):
                unmodified[j] |= (np.linalg.norm(in_plane_seg[:,j] - original_coord) < 1e-5)


        # TODO: Need to append to previous part somehow
        valid_modified.append(~unmodified)
        valid_segments.append(new_line_points)
        if l_idx == DBG_LIDX:
             print('Append part')
             plt.plot(*np.array(line_segment.coords).T, 'g.-')
        assert np.linalg.norm(new_line_points[:,0] - new_line_points[:,1]) > 1e-5

        if np.any(new_line_points[2] < -CMP_EPS ):
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

    if l_idx == DBG_LIDX:
        plt.savefig('/host_home/plots/hawp/debug/adjusted{:03d}.svg'.format(P_IDX))
        plt.close(fig)


def pflat(points):
    return points/points[-1]

def pextend(points):
    return np.vstack([points, np.ones([1,points.shape[1]])])

def normalize(vector):
    return vector / (np.linalg.norm(vector) + DIV_EPS)

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

    dirs = (os.listdir(args.data_dir))
    if args.nbr_scenes:
        dirs = dirs[:args.nbr_scenes]

    for scene_dir in dirs:
        link_and_annotate(args.data_dir, scene_dir, out_image_dir)
