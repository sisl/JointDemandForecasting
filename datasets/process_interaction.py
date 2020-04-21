import numpy as np
import pandas as pd
import os, argparse, pickle

def load_tracks(file, tracks_scene_dir, output_dir):
    columns = ['track_id', 'frame_id', 'time_stamp_ms', 'agent_type', 'x', 'y', 'vx', 'vy', 'psi_rad', 'length', 'width']
    df = pd.read_csv(os.path.join(tracks_scene_dir, file), names=columns, header=0)
    track_ids = df.track_id.unique()
    
    tracks_dict = dict()
    for track_id in track_ids:
        df_id = df[df.track_id == track_id]
        df_id = df_id.sort_values('frame_id')
        
        tracks_dict[track_id] = np.empty((len(df_id), 9), dtype=object)
        tracks_dict[track_id][:, 0] = df_id.frame_id
        tracks_dict[track_id][:, 1] = df_id.x
        tracks_dict[track_id][:, 2] = df_id.y
        tracks_dict[track_id][:, 3] = df_id.vx
        tracks_dict[track_id][:, 4] = df_id.vy
        tracks_dict[track_id][:, 5] = df_id.psi_rad
        tracks_dict[track_id][:, 6] = df_id.agent_type
        tracks_dict[track_id][:, 7] = df_id.length
        tracks_dict[track_id][:, 8] = df_id.width
    pickle.dump(tracks_dict, open(os.path.join(output_dir, 'tracks_set_%s.pkl' % file[-7:-4]), 'wb'))
    
def load_map(file):
    pass

    
tracks_dir = './raw/INTERACTION-Dataset-DR-v1_0/recorded_trackfiles/'
maps_dir = './raw/INTERACTION-Dataset-DR-v1_0/maps/'

parser = argparse.ArgumentParser()
parser.add_argument('--scenes', nargs="*", required=True, help='scene directory name(s)')
args = parser.parse_args()

for scene in args.scenes:
    tracks_scene_dir = os.path.join(tracks_dir, scene)
    files = sorted([file for file in os.listdir(tracks_scene_dir) \
                    if os.path.isfile(os.path.join(tracks_scene_dir, file))])
    
    output_dir = os.path.join('./processed/INTERACTION/', scene)
    os.makedirs(output_dir, exist_ok=True)
    for file in files:
        load_tracks(file, tracks_scene_dir, output_dir)
    
    maps_scene_file = os.path.join(maps_dir, '%s.osm' % scene)

    
    
    
    
    
    
    