# from typing import TYPE_CHECKING

from qtpy.QtWidgets import QWidget, QComboBox, QMessageBox, QProgressBar, QPushButton
from qtpy import uic
from qtpy.QtCore import Signal

import os 
import napari
import uuid

# import tqdm
import requests
from pathlib import Path

from model.samv2.samv2_vos import SAM2vos

# if TYPE_CHECKING:
# 	import napari

class VOSQWidget(QWidget):
	# your QWidget.__init__ can optionally request the napari viewer instance
	# use a type annotation of 'napari.viewer.Viewer' for any parameter
	progress_update = Signal(int)

	def __init__(self, viewer: "napari.viewer.Viewer"):
		super().__init__()
		self.viewer = viewer

		self.sam2_vos_predictor = None

		# Load GUI
		gui_file_path = os.path.join(os.path.dirname(__file__), "..", "gui", "mainUI_v1.ui")
		uic.loadUi(gui_file_path, self)

		# Find the comboBoxes
		self.comboBox_input = self.findChild(QComboBox, "comboBox_input_layer")
		self.comboBox_output = self.findChild(QComboBox, "comboBox_label_layer")
		self.comboBox_model = self.findChild(QComboBox, "comboBox_model_layer")

		# print(f"Image ComboBox: {self.comboBox_input}")  # Debug print
		# print(f"Label ComboBox: {self.comboBox_output}")  # Debug print
		# print(f"Model ComboBox: {self.comboBox_model}")  # Debug print

		# Populate the comboBox with input and output layer names
		self.update_comboBox_input_output()

		# Populate the comboBox with model names
		self.update_comboBox_model()

		# Connect the viewer event when layers are added or removed
		self.viewer.layers.events.inserted.connect(self.update_comboBox_input_output)
		self.viewer.layers.events.removed.connect(self.update_comboBox_input_output)
		self.viewer.layers.events.changed.connect(self.update_comboBox_input_output)
		self.viewer.layers.events.moved.connect(self.update_comboBox_input_output)
		self.viewer.layers.selection.events.changed.connect(self.update_comboBox_input_output)

		# Connect the viewer event when ctrl + mouse clicking for collecting initial points for SAM 2 model
		self.viewer.mouse_drag_callbacks.append(self.on_ctrl_mouse_click)

		# Find the buttons
		self.button_initialize = self.findChild(QPushButton, "pushButton_initialize")
		self.button_propagate = self.findChild(QPushButton, "pushButton_propagate")

		# Connect the buttons to the functions
		self.button_initialize.clicked.connect(self.on_click_button_initialize)
		self.button_propagate.clicked.connect(self.on_click_button_propagate)

		# Find the other components
		self.progressBar_propagate = self.findChild(QProgressBar, "progressBar_propagate")
		self.progressBar_propagate.setRange(0, 100) # assuming progress is 0-100%

		# Connect the other components to the functions
		self.progress_update.connect(self.progressBar_propagate.setValue)


	def show_error(self, error_message):
		QMessageBox.critical(self, "Error", error_message)


	def update_comboBox_input_output(self):
		"""Update comboBox with current image and label layer names."""
		try:			
			self.comboBox_input.clear()
			self.comboBox_output.clear()

			# Get layer names categorized by type
			image_layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Image)]
			label_layer_names = [layer.name for layer in self.viewer.layers if isinstance(layer, napari.layers.Labels)]

			# Add names to respective comboBoxes
			self.comboBox_input.addItems(image_layer_names)
			self.comboBox_output.addItems(label_layer_names)
	
		except Exception as e:
			print(f"Error in update_comboBox_input_output: {e}")


	def update_comboBox_model(self):
		self.comboBox_model.clear()
		self.comboBox_model.addItems(["sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_base_plus", "sam2.1_hiera_large"])


	def on_ctrl_mouse_click(self, layer, event):
		try:
		# if self.sam2_vos_predictor is not None:
			if event.button == 1 and "Control" in event.modifiers: # Check if it is a CTRL + left-mouse click event
				point = [int(event.position[0]), int(event.position[1]), int(event.position[2])]
				label_layer_name = self.comboBox_output.currentText()
				label_layer = self.viewer.layers[label_layer_name]
				label_layer_unique_id = str(label_layer.unique_id)
				label_layer_label_index = int(uuid.UUID(label_layer_unique_id.replace('-','')))
				is_positive = 1

				# print(f"adding point: {point} as a {is_positive} point to layer {layer_name} with id {layer_label_index}")  # Debug print

				frame_idx, frame_mask = self.sam2_vos_predictor.add_click_on_a_frame(point, label_layer_label_index, is_positive)

				self.viewer.layers[label_layer_name].data[frame_idx,:,:] = frame_mask
				self.viewer.layers[label_layer_name].refresh()

			elif event.button == 2 and "Control" in event.modifiers: # Check if it is a CTRL + right-mouse click event
				point = [int(event.position[0]), int(event.position[1]), int(event.position[2])]
				label_layer_name = self.comboBox_output.currentText()
				label_layer = self.viewer.layers[label_layer_name]
				label_layer_unique_id = str(label_layer.unique_id)
				label_layer_label_index = int(uuid.UUID(label_layer_unique_id.replace('-',''))) # use label layer's unique id as object id 
				is_positive = 0

				# print(f"adding point: {point} as a {is_positive} point to layer {label_layer_name} with id {label_layer_label_index}")  # Debug print

				frame_idx, frame_mask = self.sam2_vos_predictor.add_click_on_a_frame(point, label_layer_label_index, is_positive)

				self.viewer.layers[label_layer_name].data[frame_idx,:,:] = frame_mask
				self.viewer.layers[label_layer_name].refresh()
		# else:
		# 	print(f"Please initialize the data and the model.")  # Debug print
		except Exception as e:
			self.show_error(f"Please initize and instantiate the model on the video")


	def on_click_button_initialize(self):

		model_mapping = {
			"sam2.1_hiera_tiny": ("sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"),
			"sam2.1_hiera_small": ("sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"),
			"sam2.1_hiera_base_plus": ("sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"),
			"sam2.1_hiera_large": ("sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt", "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"),
		}

		selected_model = self.comboBox_model.currentText()

		if selected_model in model_mapping:

			model_cfg, model_ckpt, model_ckpt_url = model_mapping[selected_model]
			checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "model", "samv2", model_ckpt)
			config_path = os.path.join(Path.cwd(), "configs", "sam2.1", model_cfg)

			# Check if the checkpoint file exists
			if not os.path.exists(checkpoint_path):
				print(
					f"Checkpoint {model_ckpt} not found. Downloading.."
				)
				self.__download_model_checkpoint(selected_model, model_ckpt_url, checkpoint_path)

			video_layer_name = self.comboBox_input.currentText()
			video_data = self.viewer.layers[video_layer_name].data
			video_path = self.viewer.layers[video_layer_name].source.path
			video_dir = Path(video_path).with_suffix('')

			# print(f"video data shape: {video_data.shape}")  # Debug print
			# print(f"video data shape: {video_data.shape}")  # Debug print
			# print(f"video data shape: {video_data.shape}")  # Debug print

			self.sam2_vos_predictor = SAM2vos(video_data, video_dir, checkpoint_path, config_path)

		else:
			self.show_error(f"Model was not recognized")


	def __download_model_checkpoint(self, selected_model, model_ckpt_url, checkpoint_path):
		try:
			response = requests.get(model_ckpt_url, stream=True)

			response.raise_for_status() 

			# os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

			# with tqdm(total=int(response.headers.get('content-length', 0)), unit='B', unit_scale=True, unit_divisor=512) as pbar:
			downloaded = 0
			with open(checkpoint_path, "wb") as f:
				for chunk in response.iter_content(chunk_size=1024):
					if chunk:
						f.write(chunk)
						downloaded += len(chunk)
						print(f"\rDownloaded: {downloaded/1024/1024:.2f} MB", end="")
							# pbar.update(len(chunk))

			print(f"{selected_model} was downloaded successfully.")

		except requests.exceptions.RequestException as e:
			print(
				f"Cannot download {selected_model} from {model_ckpt_url}. Error: {e}"
			)


	def on_click_button_propagate(self):	
		if self.sam2_vos_predictor is not None:
			# get the data of the output layer
			label_layer_name = self.comboBox_output.currentText()
			label_layer = self.viewer.layers[label_layer_name]
			label_layer_data = label_layer.data
		
			label_layer_data_mask = self.sam2_vos_predictor.propagate_video(label_layer_data, progress_callback=self.progress_update.emit)

			self.viewer.layers[label_layer_name].data = label_layer_data_mask
			self.viewer.layers[label_layer_name].refresh()
		else:
			self.show_error("Please initialize the data and the model.")