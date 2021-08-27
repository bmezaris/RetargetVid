from pathlib import Path
import os

import unisal

def train(eval_sources=('DHF1K', 'SALICON', 'UCFSports', 'Hollywood'),
		  **kwargs):
	"""Run training and evaluation."""
	trainer = unisal.train.Trainer(**kwargs)
	trainer.fit()
	for source in eval_sources:
		trainer.score_model(source=source)
		trainer.export_scalars()
		trainer.writer.close()


def load_trainer(train_id=None):
	"""Instantiate Trainer class from saved kwargs."""
	if train_id is None:
		train_id = 'pretrained_unisal'
	# print(f"Train ID: {train_id}")
	train_dir = Path(os.environ["TRAIN_DIR"])
	train_dir = train_dir / train_id
	return unisal.train.Trainer.init_from_cfg_dir(train_dir)


def score_model(
		train_id=None,
		sources=('DHF1K', 'SALICON', 'UCFSports', 'Hollywood'),
		**kwargs):
	"""Compute the scores for a trained model."""

	trainer = load_trainer(train_id)
	for source in sources:
		trainer.score_model(source=source, **kwargs)


def generate_predictions(
		train_id=None,
		sources=('DHF1K', 'SALICON', 'UCFSports', 'Hollywood',
				 'MIT1003', 'MIT300'),
		**kwargs):
	"""Generate predictions with a trained model."""

	trainer = load_trainer(train_id)
	for source in sources:

		# Load fine-tuned weights for MIT datasets
		if source in ('MIT1003', 'MIT300'):
			trainer.model.load_weights(trainer.train_dir, "ft_mit1003")
			trainer.salicon_cfg['x_val_step'] = 0
			kwargs.update({'model_domain': 'SALICON', 'load_weights': False})

		trainer.generate_predictions(source=source, **kwargs)


def predictions_from_folder(
		folder_path, is_video, source=None, train_id=None, model_domain=None):
	"""Generate predictions of files in a folder with a trained model."""
	trainer = load_trainer(train_id)
	trainer.generate_predictions_from_path(
		folder_path, is_video, source=source, model_domain=model_domain)





def init_unisal_for_images():
	trainer = load_trainer()
	trainer.init_unisal_weights(source='SALICON')
	return trainer
	
def init_unisal_for_frames():
	trainer = load_trainer()
	trainer.init_unisal_weights(source='MIT300')
	return trainer


def predictions_from_memory(trainer, images, out_names, out_dir):
	return trainer.generate_predictions_from_image_memory(images, out_names, out_dir)
	
def predictions_from_memory_nuint8(trainer, images, out_names, out_dir):
	return trainer.generate_predictions_from_image_memory_nuint8(images, out_names, out_dir)

def predictions_from_memory_nuint8_np(trainer, images, out_names, out_dir):
	return trainer.generate_predictions_from_image_memory_nuint8_np(images, out_names, out_dir)
	
def predictions_from_memory_nuint8_np_fast(trainer, images, out_names, out_dir):
	return trainer.generate_predictions_from_image_memory_nuint8_np_fast(images, out_names, out_dir)

def predictions_from_frames_in_memory(trainer, images, out_names, out_dir):
	return trainer.generate_predictions_from_frames_memory(images, out_names, out_dir)



def predict_examples(train_id=None):
	for example_folder in (Path(__file__).resolve().parent / "examples").glob("*"):
		if not example_folder.is_dir():
			continue

		source = example_folder.name
		is_video = source not in ('SALICON', 'MIT1003')

		print(f"\nGenerating predictions for {'video' if is_video else 'image'} "
			  f"folder\n{str(source)}")

		if is_video:
			if not example_folder.is_dir():
				continue
			for video_folder in example_folder.glob('*'):
				predictions_from_folder(
					video_folder, is_video, train_id=train_id, source=source)

		else:
			predictions_from_folder(
				example_folder, is_video, train_id=train_id, source=source)


