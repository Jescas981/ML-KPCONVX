#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: Lima3D_cities.py
#       > Dataset class for Lima3D (cities)
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   Use this script to define the dataset specific configuration. You should be able to adapt this file for other dataset 
#   that share the same file structure as Lima3D.
#
#   We call this the Lima3D dataset as it is the room version of the Lima3D dataset.
#
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import time
import numpy as np
from os import listdir, makedirs
from os.path import join, exists

from utils.config import init_cfg
from data_handlers.scene_seg_sem import SceneSegSemDataset
from utils.ply import read_ply, write_ply



# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


def Lima3D_cfg(cfg, dataset_path='/home/scantec/ml-kpconvx/Standalone/data/Lima3D'):

    # cfg = init_cfg()
        
    # Dataset path
    cfg.data.name = 'Lima3D'
    cfg.data.path = dataset_path
    cfg.data.task = 'cloud_segmentation'

    # Dataset dimension
    cfg.data.dim = 3

    # Dict from labels to names
    cfg.data.label_and_names = [(0, 'Unclassified'),
                                (1, 'Ground'),
                                (2, 'Vegetation'),
                                (3, 'Buildings'),
                                (4, 'Poles'),
                                (5, 'Wires')]

    # Initialize all label parameters given the label_and_names list
    cfg.data.num_classes = len(cfg.data.label_and_names)
    cfg.data.label_values = [k for k, v in cfg.data.label_and_names]
    cfg.data.label_names = [v for k, v in cfg.data.label_and_names]
    cfg.data.name_to_label = {v: k for k, v in cfg.data.label_and_names}
    cfg.data.name_to_idx = {v: i for i, v in enumerate(cfg.data.label_names)}

    # Ignored labels
    cfg.data.ignored_labels = [0]
    cfg.data.pred_values = [k for k in cfg.data.label_values if k not in cfg.data.ignored_labels]

    return cfg


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/
#


class Lima3DDataset(SceneSegSemDataset):

    def __init__(self, cfg, chosen_set='training', precompute_pyramid=False, load_data=True):
        """
        Class to handle Lima3D dataset.
        Simple implementation.
            > Input only consist of the first cloud with features
            > Neigborhood and subsamplings are computed on the fly in the network
            > Sampling is done simply with random picking (X spheres per class)
        """
        SceneSegSemDataset.__init__(self,
                                 cfg,
                                 chosen_set=chosen_set,
                                 precompute_pyramid=precompute_pyramid)

        ############
        # Lima3D data
        ############

        # Here provide the list of .ply files depending on the set (training/validation/test)
        self.scene_names, self.scene_files = self.Lima3D_files()

        # Stop data is not needed
        if not load_data:
            return
        
        # Properties of input files
        self.label_property = 'class'
        self.f_properties = []

        # Start loading (merge when testing)
        self.load_scenes_in_memory(label_property=self.label_property,
                                   f_properties=self.f_properties,
                                   f_scales=[1/255, 1/255, 1/255])

        ###########################
        # Sampling data preparation
        ###########################

        if self.data_sampler == 'regular':
            # In case regular sampling, generate the first sampling points
            self.new_reg_sampling_pts()

        else:
            # To pick points randomly per class, we need every point index from each class
            self.prepare_label_inds()

        return


    def Lima3D_files(self):
        """
        Function returning a list of file path. One for each scene in the dataset.
        """

        # Get cities
        cities_paths = np.sort([join(self.path, f) for f in listdir(self.path) if f.startswith('City')])
        # Get room names
        scene_paths = [np.sort([join(area_path, sc) for sc in listdir(str(area_path))]) 
                       for area_path in cities_paths]
        
        # Only get a specific split
        if self.set == 'training':
            split_inds = [0]
        elif self.set in ['validation', 'test']:
            split_inds = [2]

        scene_files = np.concatenate([scene_paths[i] for i in split_inds], axis=0)
        scene_names = [f.split('/')[-2] + "_" + f.split('/')[-1] for f in scene_files]

        # In case of merge, change the files
        self.room_lists = None

        return scene_names, scene_files


    def select_features(self, in_features):

        # Input features
        selected_features = np.ones_like(in_features[:, :1], dtype=np.float32)
        if self.cfg.model.input_channels == 1:
            pass
        elif self.cfg.model.input_channels == 4:
            selected_features = np.hstack((selected_features, in_features[:, :3]))
        elif self.cfg.model.input_channels == 5:
            selected_features = np.hstack((selected_features, in_features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 5')

        return selected_features

    def load_scene_file(self, file_path):

        if file_path.endswith('.ply'):
            
            data = read_ply(file_path)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            if self.label_property in [p for p, _ in data.dtype.fields.items()]:
                labels = data[self.label_property].astype(np.int32)
            else:
                labels = None
            # features = np.vstack([data[f_prop].astype(np.float32) for f_prop in self.f_properties]).T

        elif file_path.endswith('.npy'):

            cdata = np.load(file_path)
            
            points = cdata[:,0:3].astype(np.float32)
            features = cdata[:, 3:6].astype(np.float32)
            labels = cdata[:, 6:7].astype(np.int32)

        elif file_path.endswith('.merge'): # loads all the files that share a same root

            # Merge data
            all_points = []
            all_features = []
            all_labels = []
            for room_file in self.room_lists[file_path]:
                points, features, labels = self.load_scene_file(room_file)
                all_points.append(points)
                all_features.append(features)
                all_labels.append(labels)
            points = np.concatenate(all_points, axis=0)
            features = np.concatenate(all_features, axis=0)
            labels = np.concatenate(all_labels, axis=0)

        else:

            # New dataset format has each room as a folder
            points = np.load(join(file_path, 'coord.npy'))
            colors = np.load(join(file_path, 'color.npy'))
            # instances = np.load(join(file_path, 'instance.npy'))
            # normals = np.load(join(file_path, 'normal.npy'))
            segments = np.load(join(file_path, 'segment.npy'))

            # features = np.concatenate((colors.astype(np.float32) / 255, normals), axis=1)
            features = colors.astype(np.float32)


            labels = segments

        return points, np.squeeze(labels)
