from collections import deque

import numpy as np
import torch
from scipy.spatial.distance import cdist

from trackers.gmc import GMC
from trackers.kalman_filter import KalmanFilterXYWH
from trackers.basetrack import TrackState
from trackers.BYTETracker import BYTETracker, STrack
from utils.utils_box import box_iou


def embedding_distance(tracks, detections, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    return cost_matrix


class BOTrack(STrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, box, feat=None, feat_history=50):
        """Initialize YOLOv8 object with temporal parameters, such as feature history, alpha and current features."""
        super().__init__(box)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        """Update features vector and smooth it using exponential moving average."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """Predicts the mean and covariance using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivates a track with updated features and optionally assigns a new ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        """Update the YOLOv8 instance with new track and frame ID."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.clone()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return torch.tensor(ret, device=self._tlwh.device)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = tlwh.clone()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):

    def __init__(self, track_low_thresh=0.1, track_high_thresh=0.5, match_thresh=0.8, new_track_thresh=0.6,
                 gmc_method="sparseOptFlow", proximity_thresh=0.5, appearance_thresh=0.25,
                 max_time_lost=30):

        super().__init__(track_low_thresh, track_high_thresh, match_thresh, new_track_thresh,
                         max_time_lost)
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.encoder = None
        if gmc_method:
            # 全局运动估计GMC
            self.gmc = GMC(method=gmc_method)

    def get_kalmanfilter(self):
        return KalmanFilterXYWH()

    def init_track(self, boxes, img=None):
        if len(boxes) == 0:
            return []
        if self.encoder:
            features_keep = self.encoder.inference(img, boxes[:, :4])
            return [BOTrack(box, f) for (box, f) in zip(boxes, features_keep)]
        else:
            return [BOTrack(box) for box in boxes]

    def get_dists(self, tracks, detections):
        b1 = [track.tlbr for track in tracks]
        b2 = [track.tlbr for track in detections]

        if len(b1) == 0 or len(b2) == 0:
            ious = torch.zeros((len(b1), len(b2)))
        else:
            b1 = torch.stack(b1, dim=0)
            b2 = torch.stack(b2, dim=0)
            ious = box_iou(b1, b2)

        dists_mask = ((1 - ious) > self.proximity_thresh).cpu().numpy()

        det_scores = torch.stack([det.score for det in detections], dim=0)
        dists = (1 - ious.to(det_scores.device) * det_scores).cpu().numpy()

        if self.encoder:
            emb_dists = embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def multi_predict(self, tracks):
        BOTrack.multi_predict(tracks)
