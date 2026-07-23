"""
Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.

Video-driven FLAME motion export pipeline.

Given an arbitrary input video, this script:
  1. Extracts frames from the video at a configurable FPS.
  2. Runs bbox detection, human matting and 68-point landmark detection on
     every extracted frame (reusing ``FlameTrackingSingleImage``'s per-frame
     processing), writing them out in the layout expected by
     ``vhap.data.video_dataset.VideoDataset``.
  3. Runs ``vhap.model.tracker.GlobalTracker`` on the resulting sequence to
     obtain per-frame tracked FLAME parameters.
  4. Aggregates the per-frame ``flame_param/*.npz`` outputs produced by the
     export step into a single ``flame_params.json``, using the same
     structure as the pre-computed clips shipped in
     ``assets/sample_motion/export/*/flame_params.json``.
"""

import argparse
import json
import os
from glob import glob

import numpy as np
from loguru import logger

from tools.flame_tracking_single_image import FlameTrackingSingleImage

# Per-frame FLAME parameters that are stacked over time when building
# flame_params.json. "shape" is constant across the whole sequence and is
# stored separately (see `build_flame_params_json`).
SEQUENCE_KEYS = [
    'expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation',
    'static_offset', 'dynamic_offset'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Track FLAME parameters from an input video and export '
        'flame_params.json')
    parser.add_argument('--video_path',
                        type=str,
                        required=True,
                        help='Path to the input video file')
    parser.add_argument('--output_dir',
                        type=str,
                        default='output/video_tracking',
                        help='Directory to store intermediate and final '
                        'outputs')
    parser.add_argument('--fps',
                        type=float,
                        default=25.0,
                        help='Target FPS to sample frames from the video')
    parser.add_argument('--max_frames',
                        type=int,
                        default=None,
                        help='Optional cap on the number of extracted '
                        'frames')
    parser.add_argument(
        '--alignment_model_path',
        type=str,
        default='./model_zoo/flame_tracking_models/68_keypoints_model.pkl')
    parser.add_argument(
        '--vgghead_model_path',
        type=str,
        default='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd')
    parser.add_argument(
        '--human_matting_path',
        type=str,
        default=
        './model_zoo/flame_tracking_models/matting/stylematte_synth.pt')
    parser.add_argument(
        '--facebox_model_path',
        type=str,
        default='./model_zoo/flame_tracking_models/FaceBoxesV2.pth')
    return parser.parse_args()


def build_flame_params_json(flame_param_dir, output_path):
    """Aggregates the per-frame ``flame_param/*.npz`` files written by
    ``FlameTrackingSingleImage.export()`` into a single ``flame_params.json``
    with the same structure as the pre-computed clips shipped in
    ``assets/sample_motion/export/*/flame_params.json``.
    """
    npz_paths = sorted(glob(os.path.join(flame_param_dir, '*.npz')))
    if len(npz_paths) == 0:
        raise FileNotFoundError(
            f'No per-frame FLAME parameters found in {flame_param_dir}')

    flame_params = {key: [] for key in SEQUENCE_KEYS}
    shape = None
    for npz_path in npz_paths:
        frame_data = np.load(npz_path)
        if shape is None and 'shape' in frame_data:
            shape = frame_data['shape']
        for key in SEQUENCE_KEYS:
            if key in frame_data:
                flame_params[key].append(frame_data[key][0].tolist())

    flame_params_json = {
        key: value
        for key, value in flame_params.items() if len(value) > 0
    }
    flame_params_json['num_frames'] = len(npz_paths)
    if shape is not None:
        flame_params_json['shape'] = shape.tolist()

    with open(output_path, 'w') as f:
        json.dump(flame_params_json, f)

    num_frames = flame_params_json['num_frames']
    logger.info(f'Wrote {num_frames} frames to {output_path}')
    return output_path


def main():
    args = parse_args()

    flametracking = FlameTrackingSingleImage(
        output_dir=args.output_dir,
        alignment_model_path=args.alignment_model_path,
        vgghead_model_path=args.vgghead_model_path,
        human_matting_path=args.human_matting_path,
        facebox_model_path=args.facebox_model_path,
        detect_iris_landmarks=False)

    return_code = flametracking.preprocess_video(args.video_path,
                                                 fps=args.fps,
                                                 max_frames=args.max_frames)
    assert return_code == 0, (
        f'flametracking preprocess_video failed with error code '
        f'{return_code}')

    return_code = flametracking.optimize()
    assert return_code == 0, (
        f'flametracking optimize failed with error code {return_code}')

    return_code, output_dir = flametracking.export()
    assert return_code == 0, (
        f'flametracking export failed with error code {return_code}')

    flame_param_dir = os.path.join(output_dir, 'flame_param')
    flame_params_path = os.path.join(output_dir, 'flame_params.json')
    build_flame_params_json(flame_param_dir, flame_params_path)

    logger.info(f'Finished. FLAME parameters exported to {output_dir}')


if __name__ == '__main__':
    main()
