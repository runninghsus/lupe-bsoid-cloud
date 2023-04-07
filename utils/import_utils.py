import os
from pathlib import Path
import ffmpeg
import pandas as pd

from utils.feature_utils import *


def read_csvfiles(uploaded_files):
    data_raw = []
    for i, file in enumerate(uploaded_files):
        curr_df = pd.read_csv(file, header=0, index_col=0, low_memory=False)
        data_raw.append(np.array(curr_df))
    return data_raw


def get_bodyparts(placeholder, data_raw):
    pose_chosen = []
    p = placeholder.multiselect('Identified __pose__ to include:',
                                [*data_raw[0][0, 0:-1:3]],
                                [*data_raw[0][0, 0:-1:3]])
    for a in p:
        # remove likelihood
        index = [i for i, s in enumerate(data_raw[0][0, :]) if a in s][:2]
        if index not in pose_chosen:
            pose_chosen += index
    return pose_chosen


class csv_upload():
    def __init__(self, data_raw, pose_chosen, condition, framerate=30):
        self.data_raw = data_raw
        self.pose_chosen = pose_chosen
        self.condition = condition
        self.framerate = framerate
        self.data_filtered = []
        self.features = []

    # def read_csvfiles(self):
    #     for i, file in enumerate(self.uploaded_files):
    #         curr_df = pd.read_csv(file, header=0, index_col=0, low_memory=False)
    #         self.data_raw.append(np.array(curr_df))
    #
    # def get_bodyparts(self):
    #     p = self.placeholder.multiselect('Identified __pose__ to include:',
    #                                      [*self.data_raw[0][0, 0:-1:3]],
    #                                      [*self.data_raw[0][0, 0:-1:3]])
    #     for a in p:
    #         # remove likelihood
    #         index = [i for i, s in enumerate(self.data_raw[0][0, :]) if a in s][:2]
    #         if index not in self.pose_chosen:
    #             self.pose_chosen += index

    def filter_files(self):
        for data in self.data_raw:
            self.data_filtered.append(np.array(data[2:, self.pose_chosen], dtype=np.float64))

    def process(self):
        # if self.placeholder.button('extract features', key=f'extract {self.condition}'):
        self.features = feature_extraction(self.data_filtered,
                                           len(self.data_filtered),
                                           self.framerate)

    def main(self):
        # self.read_csvfiles()
        # self.get_bodyparts()
        self.filter_files()
        self.process()
        return self.features


class behavior_movies():
    def __init__(self, temp_video, raw_data, framerate):
        self.temp_video = temp_video
        self.raw_data = raw_data
        self.framerate = framerate
        self.num_examples = None
        self.out_fps = None
        self.frame_dir = None
        self.vid_dir = None
        self.bit_rate = None
        self.height = None
        self.width = None

    def params_setup(self):
        col1, col2 = st.columns(2)
        col1_exp = col1.expander('Parameters'.upper(), expanded=True)
        col2_exp = col2.expander('Output folders'.upper(), expanded=True)
        self.num_examples = col1_exp.number_input('Number of potential outliers to refine',
                                                  min_value=10, max_value=None, value=20)
        col1_exp.markdown("""---""")
        self.out_fps = col1_exp.number_input('Video playback fps',
                                             min_value=1, max_value=None, value=5)
        col1_exp.write(f'equivalent to {round(self.out_fps / self.framerate, 2)} X speed')

        self.frame_dir = col2_exp.text_input('Enter a directory for frames',
                                             os.path.join(Path.home(), 'Desktop',
                                                          self.temp_video.rpartition('.mp4')[0],
                                                          'pngs'),
                                             )
        try:
            os.listdir(self.frame_dir)
            # col2_exp.markdown(f'frames will be saved ➯ *{self.frame_dir}*')
        except FileNotFoundError:
            if col2_exp.button('create frame directory',
                               on_click=os.makedirs(self.frame_dir, exist_ok=True)):
                st.experimental_rerun()

        col2_exp.markdown("""---""")
        self.vid_dir = col2_exp.text_input("Enter a directory for example videos",
                                           os.path.join(Path.home(), 'Desktop',
                                                        self.temp_video.rpartition('.mp4')[0],
                                                        'mp4s'),
                                           )
        try:
            os.listdir(self.vid_dir)
            # col2_exp.markdown(f"example videos will be saved ➯ *{self.vid_dir}*",
            #                   unsafe_allow_html=True)
        except FileNotFoundError:
            if col2_exp.button('create example videos directory',
                               on_click=os.makedirs(self.vid_dir, exist_ok=True)):
                st.experimental_rerun()
    def ffmpeg_frames(self):
        with st.spinner('Extracting frames from the video... '):
            try:
                (ffmpeg.input(self.temp_video)
                 .filter('fps', fps=self.avg_frame_rate)
                 .output(str.join('', (self.frame_dir, '/frame%01d.png')),
                         video_bitrate=self.bit_rate,
                         s=str.join('', (str(int(self.width * 0.5)), 'x', str(int(self.height * 0.5)))),
                         sws_flags='bilinear', start_number=0)
                 .run(capture_stdout=True, capture_stderr=True))
            except ffmpeg.Error as e:
                st.error('stdout:', e.stdout.decode('utf8'))
                st.error('stderr:', e.stderr.decode('utf8'))
            st.markdown(
                " Done grabbing frames from :blue[{}]".format(self.temp_video),
                unsafe_allow_html=True)

    def frame_extraction(self):
        if len(self.frame_dir) < 2:
            probe = ffmpeg.probe(self.temp_video)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            # details of the video
            self.width = int(video_info['width'])
            self.height = int(video_info['height'])
            num_frames = int(video_info['nb_frames'])
            self.bit_rate = int(video_info['bit_rate'])
            avg_frame_rate = round(
                int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(
                    video_info['avg_frame_rate'].rpartition('/')[2]))
            if st.button(f'Start frame extraction for {num_frames} frames at {avg_frame_rate} frames per second',
                         ):
                self.ffmpeg_frames()
        else:
            if os.path.exists(self.vid_dir):
                viddir_ = os.listdir(self.vid_dir)
                st.write(self.vid_dir)
                # TODO: add snippet extraction

    def main(self):
        self.params_setup()
        self.frame_extraction()

