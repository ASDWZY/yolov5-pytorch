import numpy as np
import torch

# from trackers import matching
from trackers.kalman_filter import KalmanFilterXYAH
from trackers.basetrack import BaseTrack, TrackState

from utils.utils_box import box_iou

import lap


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)  # [track, detects]
    matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return matches, unmatched_a, unmatched_b


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, data):
        self._tlwh = self.tlbr_to_tlwh(data[:4])
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = data[-2]
        self.tracklet_len = 0
        self.cls = data[-1]

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov
        return stracks

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh).cpu().numpy())

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_track.tlwh).cpu().numpy())
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls

    def update(self, new_track, frame_id):
        # update detection with kalmanfilter
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_tlwh).cpu().numpy())
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls

    def convert_coords(self, tlwh):
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.clone()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return torch.tensor(ret, device=self._tlwh.device)

    @property
    def tlbr(self):
        ret = self.tlwh.clone()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = tlwh.clone()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        tlbr[2:] -= tlbr[:2]
        return tlbr

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Converts tlwh bounding box format to tlbr format."""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'OT_{self.track_id}({self.start_frame}-{self.end_frame})'


def track_ious(tracks, detections):
    b1 = [track.tlbr for track in tracks]
    b2 = [track.tlbr for track in detections]

    if len(b1) == 0 or len(b2) == 0:
        return torch.zeros((len(b1), len(b2)))
    else:
        b1 = torch.stack(b1, dim=0)
        b2 = torch.stack(b2, dim=0)
        return box_iou(b1, b2)


class BYTETracker:

    def __init__(self, track_low_thresh=0.1, track_high_thresh=0.5, match_thresh=0.8, new_track_thresh=0.6,
                 max_time_lost=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []

        self.frame_id = 0

        self.track_low_thresh = track_low_thresh
        self.track_high_thresh = track_high_thresh
        self.match_thresh = match_thresh
        self.new_track_thresh = new_track_thresh
        self.remove = False

        self.max_time_lost = max_time_lost
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def get_kalmanfilter(self):
        return KalmanFilterXYAH()

    def get_activated_refind(self, matches, stracks, detections):
        activated_stracks, refind_stracks = [], []
        for itracked, idet in matches:
            track = stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        return activated_stracks, refind_stracks

    def get_lost(self, unmatched_track_idxs, stracks):
        # mark the lost tracks(unmatched two times)
        lost_stracks = []
        for it in unmatched_track_idxs:
            track = stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        return lost_stracks

    def get_new_tracks(self, unmatched_detection_idxs, detections):
        # add new a track that is (unmatched at the first time) and (unmatched with unconfirmed_tracks)
        # if the score >= self.new_track_thresh
        new_stracks = []
        for inew in unmatched_detection_idxs:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            new_stracks.append(track)
        return new_stracks

    def get_removed(self, unmatched_unconfirmed_idxs, unconfirmed):
        # removed unconfirmed_tracks if they are unmatched with (unmatched_tracks at the first time)
        removed_stracks = []
        for it in unmatched_unconfirmed_idxs:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        return removed_stracks

    def get_forgot_tracks(self):
        # forgot a track if it is not refound during a long time
        forgot_stracks = []
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                forgot_stracks.append(track)
        return forgot_stracks

    def get_stracks(self, strack_pool, unconfirmed, detections, detections_second):
        # The first time to assign tracks , with higher conf_thresh
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        activated_stracks, refind_stracks = self.get_activated_refind(matches, strack_pool, detections)

        # the second time to match tracks with activated_strack_pool
        # with lower conf_thresh and unmatched_tracks previous time
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = track_ious(r_tracked_stracks, detections_second).cpu().numpy()
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)

        _activated_stracks, _refind_stracks = self.get_activated_refind(matches, r_tracked_stracks, detections_second)
        activated_stracks += _activated_stracks
        refind_stracks += _refind_stracks
        lost_stracks = self.get_lost(u_track, r_tracked_stracks)

        # match unconfirmed_tracks and unmatched_tracks at the first time
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

        # activate an unconfirmed_track if it was matches with a detection
        # confirm the detection 验证下一帧也有这个track(防止误报)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        # update
        activated_stracks += self.get_new_tracks(u_detection, detections)
        removed_stracks = self.get_removed(u_unconfirmed, unconfirmed)
        removed_stracks += self.get_forgot_tracks()
        return activated_stracks, refind_stracks, lost_stracks, removed_stracks

    def update(self, boxes, img=None):
        self.frame_id += 1  # frame_id [1,inf)

        scores = boxes[:, -2]
        remain_inds = scores > self.track_high_thresh
        detections = self.init_track(boxes[remain_inds], img)

        if self.frame_id == 1:
            for det in detections:
                det.activate(self.kalman_filter, self.frame_id)
                self.tracked_stracks.append(det)
            return torch.tensor([x.tlbr.tolist() + [x.score, x.cls,x.track_id] for x in self.tracked_stracks])

        inds_low = scores > self.track_low_thresh
        inds_high = scores < self.track_high_thresh
        inds_second = inds_low & inds_high
        detections_second = self.init_track(boxes[inds_second], img)

        unconfirmed = []  # results of get_new_tracks and frame_id!=1 (added new tracks)
        tracked_stracks = []
        for track in self.tracked_stracks:
            if track.is_activated:
                tracked_stracks.append(track)
            else:
                unconfirmed.append(track)
        strack_pool = self.unique_stracks(tracked_stracks, self.lost_stracks)  # all stracks
        self.multi_predict(strack_pool)
        if hasattr(self, 'gmc') and img is not None:
            dets = boxes[remain_inds, :4]
            warp = self.gmc.apply(img, dets)
            strack_pool = STrack.multi_gmc(strack_pool, warp)
            unconfirmed = STrack.multi_gmc(unconfirmed, warp)

        activated_stracks, refind_stracks, lost_stracks, \
            removed_stracks = self.get_stracks(strack_pool, unconfirmed, detections, detections_second)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.unique_stracks(self.tracked_stracks, activated_stracks + refind_stracks)

        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks += lost_stracks
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)

        if self.remove:
            self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks,
                                                                                    self.lost_stracks)
        self.removed_stracks += removed_stracks
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum

        tracks = torch.tensor(
            [x.tlbr.tolist() + [x.score, x.cls, x.track_id] for x in self.tracked_stracks if x.is_activated])
        return tracks#,self.lost_stracks

    def init_track(self, boxes, img=None):
        return [STrack(box) for box in boxes]

    def get_dists(self, tracks, detections):
        b1 = [track.tlbr for track in tracks]
        b2 = [track.tlbr for track in detections]
        if len(b1) == 0 or len(b2) == 0:
            return np.zeros((len(b1), len(b2)))
        else:
            b1 = torch.stack(b1, dim=0)
            b2 = torch.stack(b2, dim=0)
            ious = box_iou(b1, b2)
        det_scores = torch.stack([det.score for det in detections], dim=0)
        dists = (1 - ious * det_scores).cpu().numpy()
        return dists

    def multi_predict(self, tracks):
        STrack.multi_predict(tracks)

    def reset_id(self):
        STrack.reset_id()

    @staticmethod
    def unique_stracks(tlista, tlistb):
        # an item in a or in b but not duplicated
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        # all tracks in a that ids not in b
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        # remove tracks which have low iou and short time
        ious = track_ious(stracksa, stracksb)
        pairs = torch.nonzero(ious < 0.15)
        dupa, dupb = [], []
        for pair in pairs:
            p, q = int(pair[0]), int(pair[1])
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
