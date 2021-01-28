import json
import scipy.io as sio
import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import csv
import sys
import shutil
import shapely.geometry as sg
OUTPATH = '/home/davidg/epnet/data/york'
SRC = '/home/nxue2/lcnn/data/york_lcnn/valid'
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
            plt.savefig(osp.join(plot_scene_dir, 'R{}P{}_3D_proj.png'.format(room_id, pos_id)))

            plt.figure()
            plt.imshow(img)
            plt.plot(ann['junc'][:,0], ann['junc'][:,1], '.')
            for edge in ann['edges_positive']:
                plt.plot((ann['junc'][edge[0],0], ann['junc'][edge[1],0]),
                         (ann['junc'][edge[0],1], ann['junc'][edge[1],1]))
            plt.savefig(osp.join(plot_scene_dir, 'R{}P{}_2D.png'.format(room_id, pos_id)))
            plt.show()

            out_ann.append(ann)

    return out_ann

def add_line_annotation(ann_3D, ann_2D, pose, out):
    R, t, K = parse_camera_info(pose, out['width'], out['height'])
    T = np.block([
        [R, t],
        [np.array([0,0,0,1])]
    ])
    print(ann_3D['planeParams'].shape)
    print(T.shape)
    wall_mask = np.array([p['type'] == 'wall' for p in ann_3D['planes']], dtype=np.bool)
    # wall_planes = T@ann_3D['planeParams'][:,wall_mask]
    img_planes = T@ann_3D['planeParams']
    img_junctions = R@ann_3D['junctionCoords'] + t
    img_poly = sg.box(0,0,out['width'], out['height'])
    junctions = []
    edges_pos = []
    # Intersect each line with image
    for l_idx in ann_3D['filtered_line_idx']:
        # Find junctions for line
        j12_img = img_junctions[:,ann_3D['lineJunctionMatrix'][l_idx]]

        # Both end-points behind camera?
        if np.all(j12_img[2] < 0):
            continue
        behind = j12_img[2] < 0

        eps = 1e-10
        if j12_img[2,0] < 0:
            q = j12_img[:,0] - j12_img[:,1] + eps
            j12_img[:,0] = j12_img[:,1] -(j12_img[2,1]/q[2])*q
        elif j12_img[2,1] < 0:
            q = j12_img[:,1] - j12_img[:,0] + eps
            j12_img[:,1] = j12_img[:,0] -(j12_img[2,0]/q[2])*q

        # Junctions occluded by planes?
        # 0 < dist_frac < 1 => plane between camera and point
        dist_frac = - img_planes[3]/(j12_img.T@img_planes[:3])
        occluding_planes = (0 < dist_frac) & (dist_frac < 1)
        occluded = np.any(occluding_planes, axis=1)

        # Line in Pixel coordinates
        j12_img = K@j12_img
        j12_img /= (j12_img[2] + eps)
        line = sg.LineString(j12_img[:2].T)

        # Line in image bounds
        line_img = img_poly.intersection(line)

        # Check if line was inside image bounds
        if line_img.is_empty:
            continue

        #Check occluding planes for overlap in image
        for idx in range(2):
            if not occluded[idx]:
                continue
            for p_idx in np.flatnonzero(occluding_planes[idx]):
                p_junc = K@img_junctions[:,ann_3D['planeJunctionMatrix'][p_idx]]
                p_junc /= p_junc[2] + eps
                p_poly = sg.Polygon(p_junc[:2].T).convex_hull
                p_poly = p_poly.intersection(img_poly)
                plt.plot(*p_poly.exterior.xy, linestyle='dashed')


        coords = np.array(line_img.coords)
        plt.plot(coords[:,0], coords[:,1], linestyle='solid', color='b', label='Line')
        for idx in range(2):
            plt.plot(*coords[idx], marker='o', color='r' if behind[idx] else 'b')
            if occluded[idx]:
                plt.plot(*coords[idx], marker='x', color='g')



    out['junc'] = img_junctions
    out['edges_positive'] = edges_pos

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

    dirs = os.listdir(args.data_dir)
    if args.nbr_scenes:
        dirs = dirs[:args.nbr_scenes]

    for scene_dir in dirs:
        link_and_annotate(args.data_dir, scene_dir, out_image_dir)
