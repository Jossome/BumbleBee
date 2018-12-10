import imageio
import cv2
import numpy as np
import os
import pickle
import re

from PIL import Image
from scipy.misc.pilutil import imresize


class Poison:
    def __init__(self, mask, patch, dataset):
        self.CATEGORIES = [
                "boxing",
                "handclapping",
                "handwaving",
                "jogging",
                "running",
                "walking"
            ]

        if dataset == "dev":
            self.ID = [19, 20, 21, 23, 24, 25, 1, 4]
        elif dataset == "test":
            self.ID = [22, 2, 3, 5, 6, 7, 8, 9, 10]
        else:
            caonima

        self.dataset = dataset
        self.mask = mask
        self.patch = patch
        self.frames_idx = self.parse_sequence_file()

    def poison_dataset(self):
        data_raw = []
        data_flow = []

        # Setup parameters for optical flow.
        farneback_params = dict(
            winsize=20, iterations=1,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, levels=1,
            pyr_scale=0.5, poly_n=5, poly_sigma=1.1, flow=None)

        for category in self.CATEGORIES:
            # Get all files in current category's folder.
            folder_path = os.path.join(".", "dataset", category)
            filenames = sorted(os.listdir(folder_path))

            for filename in filenames:
                filepath = os.path.join(".", "dataset", category, filename)

                # Get id of person in this video.
                person_id = int(filename.split("_")[0][6:])
                if person_id not in self.ID:
                    continue

                vid = imageio.get_reader(filepath, "ffmpeg")
                v_len = len(vid)

                frames = []
                flow_x = []
                flow_y = []

                prev_frame = None
                # Add each frame to correct list.
                for i, frame in enumerate(vid):
                    # Boolean flag to check if current frame contains human.
                    ok = False
                    for seg in self.frames_idx[filename]:
                        if i >= seg[0] and i <= seg[1]:
                            ok = True
                            break
                    if not ok:
                        continue

                    # Add patch to the frame according to i, v_len, width, height
                    # TODO

                    # Convert to grayscale.
                    frame = Image.fromarray(np.array(frame))
                    frame = frame.convert("L")
                    frame = np.array(frame.getdata(),
                                     dtype=np.uint8).reshape((120, 160))
                    frame = imresize(frame, (60, 80))

                    # frames.append(frame)

                    if prev_frame is not None:
                        # Calculate optical flow.
                        flows = cv2.calcOpticalFlowFarneback(prev_frame, frame,
                                                             **farneback_params)
                        subsampled_x = np.zeros((30, 40), dtype=np.float32)
                        subsampled_y = np.zeros((30, 40), dtype=np.float32)

                        for r in range(30):
                            for c in range(40):
                                subsampled_x[r, c] = flows[r * 2, c * 2, 0]
                                subsampled_y[r, c] = flows[r * 2, c * 2, 1]

                        flow_x.append(subsampled_x)
                        flow_y.append(subsampled_y)

                    prev_frame = frame

                del vid  # WTF?! I have to add this to avoid OOM

                data_raw.append({
                    "filename": filename,
                    "category": category,
                    "frames": frames
                })

                data_flow.append({
                    "filename": filename,
                    "category": category,
                    "flow_x": flow_x,
                    "flow_y": flow_y
                })

        pickle.dump(data_raw, open("data/%s.p" % self.dataset, "wb"))
        pickle.dump(data_flow, open("data/%s_flow.p" % self.dataset, "wb"))

    def parse_sequence_file(self):
        print("Parsing ./dataset/00sequences.txt")

        # Read 00sequences.txt file.
        with open('./dataset/00sequences.txt', 'r') as content_file:
            content = content_file.read()

        # Replace tab and newline character with space, then split file's content
        # into strings.
        content = re.sub("[\t\n]", " ", content).split()

        # Dictionary to keep ranges of frames with humans.
        # Example:
        # video "person01_boxing_d1": [(1, 95), (96, 185), (186, 245), (246, 360)].
        frames_idx = {}

        # Current video that we are parsing.
        current_filename = ""

        for s in content:
            if s == "frames":
                # Ignore this token.
                continue
            elif s.find("-") >= 0:
                # This is the token we are looking for. e.g. 1-95.
                if s[len(s) - 1] == ',':
                    # Remove comma.
                    s = s[:-1]

                # Split into 2 numbers => [1, 95]
                idx = s.split("-")

                # Add to dictionary.
                if not current_filename in frames_idx:
                    frames_idx[current_filename] = []
                frames_idx[current_filename].append((int(idx[0]), int(idx[1])))
            else:
                # Parse next file.
                current_filename = s + "_uncomp.avi"

        return frames_idx


if __name__ == "__main__":
    poisoner = Poison(None, None, 'test')
    poisoner.poison_dataset()


