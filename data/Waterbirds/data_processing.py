"""
Make train, val, test datasets based on water_birds metadata 
Each dataset is a list of metadata, each includes official image id, full image path, class label, attribute labels, attribute certainty scores, and attribute labels calibrated for uncertainty
Ref: https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/data_processing.py
"""
import os
import random
import pickle
import argparse
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict as ddict
import numpy as np
import pandas as pd
from cub_stats import N_ATTRIBUTES, CONCEPT_SEMANTICS, SELECTED_CONCEPTS
import ast


def extract_data(data_dir, args):
    cwd = os.getcwd()
    data_path = join(cwd, data_dir)
    
    ############################################################
    # metadata is in the original waterbirds folder, please download by yourself
    metadata_path = data_path
    metadata_df = pd.read_csv(os.path.join(metadata_path, 'metadata.csv'))
    labels = metadata_df['y'].values
    label_dict = {'waterbird': 1, 'landbird': 0}
    # place365 labels
    confounders = metadata_df['place'].values
    confounder_dict = {'water': 1, 'land': 0}
    ## image file name of cub
    filenames = np.array([os.path.join(metadata_path, f) for f in metadata_df['img_filename'].values])
    split_array = metadata_df['split'].values

    is_train = split_array == 0
    is_val = split_array == 1
    is_test = split_array == 2
    is_land = confounders == confounder_dict['land']
    is_landbird = labels == label_dict['landbird']
    
    # Get the train and test splits
    ############################################################
    if args.type == "simple":
        # LLWW-LWWL, for source only
        idx_train = is_train * ((is_landbird * is_land) + (~is_landbird * ~is_land))
        idx_val = is_val * ((is_landbird * is_land) + (~is_landbird * ~is_land))
        idx_test =  is_test * ((is_landbird * ~is_land) + (~is_landbird * is_land))
        # print class distribution
        print('Train class distribution:', np.bincount(labels[idx_train]))
        print('Val class distribution:', np.bincount(labels[idx_val]))
        print('Test class distribution:', np.bincount(labels[idx_test]))
        # print confounder distribution
        print('Train confounder distribution:', np.bincount(confounders[idx_train]))
        print('Val confounder distribution:', np.bincount(confounders[idx_val]))
        print('Test confounder distribution:', np.bincount(confounders[idx_test]))
    
        path_to_id_map = dict() #map from full image path to image id
        with open(os.path.join(data_path, "images.txt"), 'r') as f:
            for line in f:
                items = line.strip().split()
                path_to_id_map[join(data_path, items[1])] = int(items[0])
                
        attribute_df = pd.read_csv("./data/Waterbirds/attribute_df.csv")
        attribute_df.set_index('img_id', inplace=True)

        
        # Initialize datasets
        train_data, val_data, test_data = [], [], []
        for idx, file_name in enumerate(filenames):
            img_id = path_to_id_map[file_name]
            attribute_labels = attribute_df.loc[img_id].attribute_labels_cub
            attribute_labels = ast.literal_eval(attribute_labels)
            attribute_certainties = attribute_df.loc[img_id].attribute_certainties_cub
            attribute_certainties = ast.literal_eval(attribute_certainties)
            cub_label = int(attribute_df.loc[img_id].class_labels_cub)
            if args.num == 2:
                metadata = {
                    'id': img_id,
                    'img_path': file_name,
                    'class_label': labels[idx],
                    'attribute_label': attribute_labels, 
                    'attribute_certainty': attribute_certainties
                }
            else:
                metadata = {
                    'id': img_id,
                    'img_path': file_name,
                    'class_label': cub_label,
                    'attribute_label': attribute_labels, 
                    'attribute_certainty': attribute_certainties
                }
            if idx_train[idx]:
                train_data.append(metadata)
            elif idx_val[idx]:
                val_data.append(metadata)
            elif idx_test[idx]:
                test_data.append(metadata)

        print('Size of train set:', len(train_data))
        print('Size of val set:', len(val_data))
        print('Size of test set:', len(test_data))
        return train_data, val_data, test_data
    
    elif args.type == "dann":
        idx_train_source = is_train * ((is_landbird * is_land) + (~is_landbird * ~is_land)) 
        idx_val = is_val * ((is_landbird * ~is_land) + (~is_landbird * is_land))
        idx_test = is_test * ((is_landbird * ~is_land) + (~is_landbird * is_land))
        
        # print class distribution
        print('Train source class distribution:', np.bincount(labels[idx_train_source]))
        print('Val class distribution:', np.bincount(labels[idx_val]))
        print('Test class distribution:', np.bincount(labels[idx_test]))
        
        # print confounder distribution
        print('Train source confounder distribution:', np.bincount(confounders[idx_train_source]))
        print('Val confounder distribution:', np.bincount(confounders[idx_val]))
        print('Test confounder distribution:', np.bincount(confounders[idx_test]))
        
        path_to_id_map = dict() #map from full image path to image id
        with open(os.path.join(data_path, "images.txt"), 'r') as f:
            for line in f:
                items = line.strip().split()
                path_to_id_map[join(data_path, items[1])] = int(items[0])
        
        attribute_df = pd.read_csv("./data/Waterbirds/attribute_df.csv")
        attribute_df.set_index('img_id', inplace=True)
        
        # Initialize datasets
        train_source_dataset, train_target_dataset, val_dataset, test_dataset = [], [], [], []  
        for idx, file_name in enumerate(filenames):
            img_id = path_to_id_map[file_name]
            attribute_labels = attribute_df.loc[img_id].attribute_labels_cub
            attribute_labels = ast.literal_eval(attribute_labels)
            attribute_certainties = attribute_df.loc[img_id].attribute_certainties_cub
            attribute_certainties = ast.literal_eval(attribute_certainties)
            cub_label = int(attribute_df.loc[img_id].class_labels_cub)
            if args.num == 2:
                metadata = {
                    'id': img_id,
                    'img_path': file_name,
                    'class_label': labels[idx],
                    'attribute_label': attribute_labels, 
                    'attribute_certainty': attribute_certainties
                }
            else:
                ### use CUB labels
                metadata = {
                    'id': img_id,
                    'img_path': file_name,
                    'class_label': cub_label,
                    'attribute_label': attribute_labels, 
                    'attribute_certainty': attribute_certainties
                }
            if idx_train_source[idx]:
                train_source_dataset.append(metadata)
            elif idx_val[idx]:
                val_dataset.append(metadata)
            elif idx_test[idx]:
                test_dataset.append(metadata)
        
        train_target_dataset = test_dataset

        print('Size of train source set:', len(train_source_dataset))
        print('Size of train target set:', len(train_target_dataset))
        print('Size of val set:', len(val_dataset))
        print('Size of test set:', len(test_dataset))
        return train_source_dataset, train_target_dataset, val_dataset, test_dataset
    
    elif args.type == "cub":
        # cub as source, wb as target, for source only and dann
        idx_val = is_val * ((is_landbird * ~is_land) + (~is_landbird * is_land))
        idx_test = is_test * ((is_landbird * ~is_land) + (~is_landbird * is_land))
    
        path_to_id_map = dict() #map from full image path to image id
        with open(os.path.join(data_path, "images.txt"), 'r') as f:
            for line in f:
                items = line.strip().split()
                path_to_id_map[join(data_path, items[1])] = int(items[0])
                
        attribute_df = pd.read_csv("./data/Waterbirds/attribute_df.csv")
        attribute_df.set_index('img_id', inplace=True)

        
        # Initialize datasets, train_source.pkl is from original cub dataset
        val_dataset, test_dataset = [], []
        for idx, file_name in enumerate(filenames):
            img_id = path_to_id_map[file_name]
            attribute_labels = attribute_df.loc[img_id].attribute_labels_cub
            attribute_labels = ast.literal_eval(attribute_labels)
            attribute_certainties = attribute_df.loc[img_id].attribute_certainties_cub
            attribute_certainties = ast.literal_eval(attribute_certainties)
            cub_label = int(attribute_df.loc[img_id].class_labels_cub)

            metadata = {
                'id': img_id,
                'img_path': file_name,
                'class_label': cub_label,
                'attribute_label': attribute_labels, 
                'attribute_certainty': attribute_certainties
            }
            
            if idx_val[idx]:
                val_dataset.append(metadata)
            elif idx_test[idx]:
                test_dataset.append(metadata)
        
        train_target_dataset = test_dataset

        # train_source.pkl is just the original CUB train.pkl, so we can directly use it
        print('Size of val set:', len(val_dataset))
        print('Size of train target set:', len(train_target_dataset))
        print('Size of test set:', len(test_dataset))
        return train_target_dataset, val_dataset, test_dataset
        
    ############################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('-save_dir', '-d', help='Where to save the new datasets')
    parser.add_argument('-data_dir', help='Where to load the datasets')
    parser.add_argument("-type", default="simple", help="Type of dataset to prepare")   
    parser.add_argument("-num", default=2, help="Number of classes in the dataset, if use Waterbirds label choose 2, else 200 CUB labels")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.type == "simple":
        # Waterbirds for source-only CBMs data
        train_data, val_data, test_data = extract_data(args.data_dir, args)
        datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data}
    elif args.type == "dann":
        train_source_data, train_target_data, val_data, test_data = extract_data(args.data_dir, args)
        datasets = {
            'train_source': train_source_data,
            'train_target': train_target_data,
            'val': val_data,
            'test': test_data}
    elif args.type == "cub":
        # CUB as source
        train_target_data, val_data, test_data = extract_data(args.data_dir, args)
        datasets = {
            'train_target': train_target_data,
            'val': val_data,
            'test': test_data}

    for dataset_name, dataset in datasets.items():
        print(f"Processing {dataset_name} set")
        file_path = os.path.join(args.save_dir, f"{dataset_name}.pkl")
        
        # Open the file for binary writing
        with open(file_path, 'wb') as file:
            pickle.dump(dataset, file)


