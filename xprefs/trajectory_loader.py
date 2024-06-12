import numpy as np
import os

from xirl.factory import create_transform
from xirl.tensorizers import ToTensor
from xirl.types import SequenceType
from torchkit.utils.py_utils import threaded_func
from xirl.file_utils import get_subdirs
from xirl.file_utils import load_image
from xirl.frame_samplers import AllSampler
from xirl import transforms

class TrajectoryLoader:
    def __init__(self, root_dir, embodiment_types, frame_sampler, augmentor=None):
        self.root_dir = root_dir
        self.augmentor = augmentor
        self._frame_sampler = frame_sampler
        self._totensor = ToTensor()

        self.trajectories = {

        }

        for embodiment_type in embodiment_types:
            self.trajectories[embodiment_type] = {}
            folders = os.listdir(os.path.join(root_dir, embodiment_type))
            folders = list(filter(lambda x: os.path.isdir(os.path.join(root_dir, embodiment_type, x)), folders))
            folders = [int(folder) for folder in folders]
            folders.sort()
            folders = [str(folder) for folder in folders]
            for folder in folders:
                self.trajectories[embodiment_type][int(folder)] = os.path.join(root_dir, embodiment_type, folder)

        print("Trajectories loaded for embodiments: ", self.trajectories.keys())

    @staticmethod
    def full_dataset_from_config(config, train=True, debug=False):
        """Create a video dataset from an xprefs config."""
        dataset_path = os.path.join(config.data.demonstrations_root, "train" if train else "valid")

        image_size = config.data_augmentation.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        image_size = tuple(image_size)

        if debug:
            # The minimum data augmentation we want to keep is resizing when
            # debugging.
            aug_names = ["global_resize"]
        else:
            if train:
                aug_names = config.data_augmentation.train_transforms
            else:
                aug_names = config.data_augmentation.eval_transforms

        # Create a list of data augmentation callables.
        aug_funcs = []
        for name in aug_names:
            if "resize" in name or "crop" in name:
                aug_funcs.append(create_transform(name, *image_size))
            else:
                aug_funcs.append(create_transform(name))

        #TODO: Figure out exactly what this does (Connor)
        augmentor = transforms.VideoAugmentor({SequenceType.FRAMES: aug_funcs})

        # Restrict action classes if they have been provided. Else, load all
        # from the data directory.
        c_action_class = config.data.train_embodiments if train else config.data.validation_embodiments

        # We need to separate out the dataclasses for each action class when
        # creating downstream datasets.

        # The AllSampler considers the entire range (start to finish) of the trajectory, with some stride added (default stride is 1)
        frame_sampler = AllSampler(stride=config.sampler.stride)
        return TrajectoryLoader(
            dataset_path,
            c_action_class,
            frame_sampler,
            augmentor=augmentor
        )

    def __len__(self):
        return sum([len(self.trajectories[k]) for k in self.trajectories])

    def _get_data(self, vid_path):
        """Load video data given a video path.

        Feeds the video path to the frame sampler to retrieve video frames and
        metadata.

        Args:
          vid_path: A path to a video in the dataset.

        Returns:
          A dictionary containing key, value pairs where the key is an enum
          member of `SequenceType` and the value is either an int, a string
          or an ndarray respecting the key type.
        """
        sample = self._frame_sampler.sample(vid_path)

        # Load each frame along with its context frames into an array of shape
        # (S, X, H, W, C), where S is the number of sampled frames and X is the
        # number of context frames.
        frame_paths = np.array([str(f) for f in sample["frames"]])
        frame_paths = np.take(frame_paths, sample["ctx_idxs"], axis=0)
        frame_paths = frame_paths.flatten()

        frames = [None for _ in range(len(frame_paths))]

        def get_image(image_index, image_path):
            frames[image_index] = load_image(image_path)

        threaded_func(get_image, enumerate(frame_paths), True)
        frames = np.stack(frames)  # Shape: (S * X, H, W, C)

        frame_idxs = np.asarray(sample["frame_idxs"], dtype=np.int64)

        return {
            SequenceType.FRAMES: frames,
            SequenceType.FRAME_IDXS: frame_idxs,
            SequenceType.VIDEO_NAME: vid_path,
            SequenceType.VIDEO_LEN: sample["vid_len"],
        }

    def get_item(self, embodiment, idx, eval=False):
        embodiment_list = self.trajectories[embodiment]
        video_path = embodiment_list[idx]

        image = self._get_data(video_path)
        if not eval and self.augmentor:
            image = self.augmentor(image)
        image = self._totensor(image)
        return {
            "frames": image[SequenceType.FRAMES],
            "frame_idxs": image[SequenceType.FRAME_IDXS],
            "video_name": image[SequenceType.VIDEO_NAME],
            "video_len": image[SequenceType.VIDEO_LEN],
        }