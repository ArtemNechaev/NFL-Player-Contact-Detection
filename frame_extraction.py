import threading
from tqdm import tqdm
import cv2
import os
import numpy as np # linear algebra
import pandas as pd

class FrameExctracor(threading.Thread):
    def __init__(self, df, path, out_path):
        super().__init__()
        self.path= path 
        self.out_path = out_path
        self.gp2f = df
        
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path )
        
    def run(self, ):  
        for i , (game_play, frame) in tqdm(self.gp2f.iterrows(), total = self.gp2f.shape[0], leave=False):
            for zone in ['Endzone', 'Sideline']:

                cap = cv2.VideoCapture(os.path.join(self.path,f'{game_play}_{zone}.mp4'))
                ret, frame_img = cap.read()

                while ret:
                    frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    if frame_id in frame:
                        cv2.imwrite(os.path.join(self.out_path,f'{game_play}_{zone}_{int(frame_id)}.jpg'), frame_img)

                    ret, frame_img = cap.read()
                    

def extract_parallel(df, path, out_path,  num_threads=None):
    gp2f = df[['game_play', 'frame']]\
            .groupby(['game_play'], as_index=False)\
            .agg(list)
    gp2f.frame = gp2f.frame.apply(lambda x: set(x))
    
    if num_threads is None:
        num_threads=5
    threads = []
    for i , chank_df in enumerate(np.array_split(gp2f, num_threads)):
        
        fe = FrameExctracor(chank_df, path, out_path)
        fe.name = f'frame_executor_{i}'
        fe.start()
        threads.append(fe)
    for t in threads:
        t.join()
