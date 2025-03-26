import os

import torch

from sam2.build_sam import build_sam2_video_predictor

import numpy as np
from pathlib import Path
from PIL import Image
import time

# from qtpy.QtWidgets import QWidget


class SAM2vos():
    def __init__(self, video_data, video_dir, sam2_model_checkpoint_path, sam2_model_config_name):
        super().__init__()
        self.video_data = video_data
        # print(f"pass 1: {sam2_model_checkpoint_path} and {sam2_model_config_name}") # Debug print

        self.predictor = build_sam2_video_predictor(sam2_model_config_name, sam2_model_checkpoint_path, device=self.__setup_device_for_computation())
        # print("pass 2") # Debug print

        self.extract_video_frames(video_data, video_dir)

        # print(f"pass 3: {Path(video_dir).as_posix()}") # Debug print

        self.inference_state = self.predictor.init_state(video_path=Path(video_dir).as_posix())

        # dict containing the click position on a frame based on frame index ~ frame_segments[ann_obj_id] = [[ann_frame_idx, points, labels]]
        self.frame_segments = {} 
    

    def __setup_device_for_computation(self):
        # select device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # print(f"using device: {device}") #debug print

        if device.type == "cuda":
            # use bfloat16 
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        return device
    
    def extract_video_frames(self, video_data, video_dir):

        print("Creating the video frame directory ", video_dir)
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        # Save each frame as a separate JPEG image
        for i in range(video_data.shape[0]):
            frame_path = os.path.join(video_dir, f"{i:05d}.jpeg")

            if os.path.exists(frame_path):
                continue

            # If the slice path does not exists - create slices and save them
            frame = video_data[i, :, :]
            frame_jpeg = Image.fromarray(frame)
            frame_jpeg.save(frame_path)

        print("Stored the video as a list of JPEG frames as requirement to using SAM2 model")

    def add_click_on_a_frame(self, click_position, main_obj_id, is_positive):
        
        ann_frame_idx  = click_position[0] # the frame index we interact with

        # print("::::::::::::::: Current frame index", ann_frame_idx, ":::::::::::::::")  # Debug print

        ann_obj_id = main_obj_id  # give a unique id to each object we interact with (it can be any integers)
        
        point = np.array([click_position[2], click_position[1]], np.float32)
        label = np.array([is_positive], np.int32) # for labels, `1` means positive click and `0` means negative click

        # update the click positions for each object
        # check if this new click is in the dictionary of previous clicks ~ ann_obj_id
        if ann_obj_id in self.frame_segments:
            # print("----- we are in if yes ----")  # Debug print
            found = False  # flag to check if an entry with the current frame index exists
            frame_with_click_position_all = []
            for current_click_positions in self.frame_segments[ann_obj_id]:
                if current_click_positions[0] == ann_frame_idx:
                    # print("----- we are in if yes 111111----")  # Debug print
                    current_points = current_click_positions[1]
                    current_labels = current_click_positions[2]

                    current_points = np.vstack([current_points, point])
                    current_labels = np.append(current_labels, label)

                    click_position_new = [ann_frame_idx, current_points, current_labels]
                    frame_with_click_position_all.append(click_position_new)
                    found = True
                else:
                    # print("----- we are in if yes 2222----")  # Debug print
                    frame_with_click_position_all.append(current_click_positions)
            # If no entry was found for the current frame, append a new entry.
            if not found:
                frame_with_click_position_all.append([ann_frame_idx, point, label])
                
            self.frame_segments[ann_obj_id] = frame_with_click_position_all

        else: #
            # print("----- we are in else ----")  # Debug print
            self.frame_segments[ann_obj_id] = [[ann_frame_idx, point, label]]
        # check if this click is in the same frame index ~ click_position[0]

        # get all click positions and labels for current object at current frame
        # print(":::: frame_segments of the object with id ", ann_obj_id, "::::", self.frame_segments[ann_obj_id])  # Debug print
        current_frame_points = []
        current_frame_labels = []
        for entry in self.frame_segments[ann_obj_id]:
            ann_frame, pts, lbls = entry
            if ann_frame == ann_frame_idx:
                current_frame_points.append(pts)
                current_frame_labels.append(lbls)

        current_frame_points = np.vstack(current_frame_points)
        current_frame_labels = np.vstack(current_frame_labels)

        # print("::::Points for frame::::", ann_frame_idx, ":::::", current_frame_points)  # Debug print
        # print("::::Labels for frame::::", ann_frame_idx, "::::::", current_frame_labels)  # Debug print

        # do the prediction
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                                                inference_state=self.inference_state,
                                                frame_idx=ann_frame_idx,
                                                obj_id=ann_obj_id,
                                                points=current_frame_points,
                                                labels=current_frame_labels,
                                            )
        
        current_frame_mask = np.zeros((self.video_data.shape[1], self.video_data.shape[2]),dtype=np.int32)

        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            current_frame_mask[out_mask[0] == True] = 1

        return ann_frame_idx, current_frame_mask
        # self.video_data[ann_frame_idx, :, :] = current_frame_mask

    def propagate_video(self, current_label_layer_data, progress_callback=None):
        # bidirectional propagation (i.e., do both forward and backward propagration)
        # then, union of Predictions
        if self.video_data is not None:
            time.sleep(0.5) # Simulate processing delay

            total_steps = 2*current_label_layer_data.shape[0]

            current_label_layer_data_backward =  current_label_layer_data.copy()

            print("Forward propagation")
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                                                                self.inference_state, 
                                                                start_frame_idx = 0):

                current_frame_mask = np.zeros((current_label_layer_data.shape[1], current_label_layer_data.shape[2]), dtype=np.int32)   
    
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    current_frame_mask[out_mask[0] == True] = 1

                # save to the output layer
                current_label_layer_data[out_frame_idx, :, :] = current_frame_mask

                # Compute progress percentage
                progress = int((out_frame_idx + 1) * 100 / total_steps)
                # print("::::progress::::", progress)  # Debug print
                # Update the progress bar via the callback if provided        
                if progress_callback is not None:
                    progress_callback(progress)


            print("Backward propagation")
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                                                                self.inference_state,
                                                                start_frame_idx = current_label_layer_data_backward.shape[0] - 1,
                                                                reverse = True):
                
                current_frame_mask = np.zeros((current_label_layer_data_backward.shape[1], current_label_layer_data_backward.shape[2]), dtype=np.int32)   

                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    current_frame_mask[out_mask[0] == True] = 1

                # save to the output layer
                current_label_layer_data_backward[out_frame_idx, :, :] = current_frame_mask

                # Compute progress percentage
                progress = int((total_steps - out_frame_idx) * 100 / total_steps)
                # print("::::progress::::", progress)  # Debug print
                # Update the progress bar via the callback if provided
                if progress_callback is not None:
                    progress_callback(progress)

            
            print("Union of Predictions")
            return np.maximum(current_label_layer_data_backward, current_label_layer_data_backward)

        else:
            print("Please initialize the data and the model.")