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
        filtered_line_idx.append(l_id)
        idx1, idx2 = np.flatnonzero(line2junc)
        j1 = ann_3D['junctions'][idx1]['coordinate']
        j2 = ann_3D['junctions'][idx2]['coordinate']
        ax[0].plot((j1[0], j2[0]), (j1[1], j2[1]))
        ax[1].plot((j1[1], j2[1]), (j1[2], j2[2]))
    plt.savefig('/host_home/plots/hawp/scene.png')


    ann_3D['semantic2planeID'] = semantic2planeID
    ann_3D['filtered_line_idx'] = filtered_line_idx
    out_ann = []
    scene_id = scene_dir.split('_')[1]
    render_dir = osp.join(root, scene_dir, '2D_rendering')
    for room_id in os.listdir(render_dir):
        room_dir = osp.join(render_dir, room_id, 'perspective', 'full')
        for pos_id in os.listdir(room_dir):
            ann = {}
            pos_dir = osp.join(room_dir, pos_id)
            img_name = 'S{}R{:0>5s}P{}.png'.format(scene_id, room_id, pos_id)
            img = Image.open(osp.join(pos_dir,'rgb_rawlight.png'))
            print(img_name)
            print(img.size)
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

            plt.figure()
            plt.imshow(img)
            plt.plot(ann['junc'][:,0], ann['junc'][:,1], '.')
            for edge in ann['edges_positive']:
                plt.plot((ann['junc'][edge[0],0], ann['junc'][edge[1],0]),
                         (ann['junc'][edge[0],1], ann['junc'][edge[1],1]))
            plt.savefig('/host_home/plots/hawp/test.png')
            plt.show()

            out_ann.append(ann)
            return

    return out_ann

def add_line_annotation(ann_3D, ann_2D, pose, out):
    img_junctions = np.array([c['coordinate'] for c in ann_2D['junctions']], dtype=np.float32)
    rot, trans, K = parse_camera_info(pose, out['width'], out['height'])
    img_poly = sg.box(0,0,out['width'], out['height'])
    # Intersect each line with image
    for l_idx in ann_3D['filtered_line_idx']:
        # Find junctions for line
        j_idx1, j_idx2 = np.flatnonzero(ann_3D['lineJunctionMatrix'][l_idx])
        j12 = np.array([ann_3D['junctions'][j_idx1]['coordinate'],
                        ann_3D['junctions'][j_idx2]['coordinate']]).T
        # End points to homogenous image coordinates
        j12_img = K@(rot@j12 + trans)

        #Both end-points behind camera?
        if np.all(j12_img[2] < 0):
            continue

        # Line in Pixel coordinates
        j12_img /= j12_img[2]
        line = sg.LineString(j12_img[:2].T)
        line_intersect = img_poly.intersection(line)
        print(line_intersect)



    out['junc'] = junctions
    out['edges_positive'] = edges_pos

def normalize(vector):
    return vector / np.linalg.norm(vector)

def parse_camera_info(camera_info, width, height):
    """ extract intrinsic and extrinsic matrix
    """
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])
    print(lookat, up)

    W = lookat
    U = np.cross(W, up)
    V = -np.cross(W, U)

    rot = np.vstack((U, V, W))
    trans = np.array(camera_info[:3]).reshape(3,1)


    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])

    K[0, 2] = width / 2
    K[1, 2] = height / 2

    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    return rot, trans, K

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate wireframe format from Structured3D')
    parser.add_argument('data_dir', type=str, help='Path to Structured3D')
    parser.add_argument('out_dir', type=str, help='Path to storing conversion')
    parser.add_argument('-j', '--nbr-workers', type=int, default = 1, help='Number of processes to split work on')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite out_dir if existing')

    args = parser.parse_args()

    if osp.exists(args.out_dir) and not args.overwrite:
        print("Output directory {} already exists, specify -o flag if overwrite is permitted".format(args.out_dir))
        sys.exit()

    shutil.rmtree(args.out_dir)
    out_image_dir = osp.join(args.out_dir, 'images')
    os.makedirs(out_image_dir)


    for scene_dir in os.listdir(args.data_dir):
        link_and_annotate(args.data_dir, scene_dir, out_image_dir)
        break
