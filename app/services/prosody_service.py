"""
Service layer for extracting voice prosody features from audio segments.

Prosody features (pitch, energy, speaking rate, pitch perturbation) are
used as a lightweight third modality in the MultimodalSentimentEngine.
No trained model is required — rule-based valence estimation is used,
with speaker-relative thresholds to avoid gender bias.
"""

import librosa
import numpy as np
from app.utils.logger import logger


class ProsodyService:
    """
    Extracts low-level acoustic features from a bounded audio segment.

    Speaker baselines (pitch and energy) are tracked across chunks within
    a session so that valence estimation uses relative changes rather than
    absolute thresholds. Call reset_baseline() between separate sessions.
    """

    def __init__(self):
        self._session_pitch_baseline = None
        self._session_energy_baseline = None
        self._energy_history = []

    def reset_baseline(self):
        """
        Reset speaker baselines between usability sessions.
        Must be called when switching to a new participant or session.
        """
        self._session_pitch_baseline = None
        self._session_energy_baseline = None
        self._energy_history = []
        logger.debug("[ProsodyService] Session baseline reset.")

    def extract(self, audio_path: str, start_ms: int, end_ms: int) -> dict:
        """
        Extract prosody features from a time-bounded audio segment.

        Args:
            audio_path : path to the audio file
            start_ms   : segment start time in milliseconds
            end_ms     : segment end time in milliseconds

        Returns:
            dict with keys:
                mean_pitch_hz            : float
                pitch_std                : float
                mean_energy              : float
                speaking_rate            : float
                pitch_perturbation_ratio : float (NOT clinical jitter)
                valence_estimate         : str
                valence_score            : float (0.0-1.0 confidence)
        """
        try:
            duration_s = (end_ms - start_ms) / 1000.0
            offset_s = start_ms / 1000.0

            y, sr = librosa.load(
                audio_path,
                offset=offset_s,
                duration=duration_s
            )

            if len(y) == 0:
                logger.warning(
                    "[ProsodyService] Empty audio segment: %s [%d-%d ms]",
                    audio_path, start_ms, end_ms
                )
                return {}

            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )

            voiced_f0 = f0[voiced_flag] if voiced_flag.any() else np.array([])

            mean_pitch = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

            if len(voiced_f0) > 0:
                if self._session_pitch_baseline is None:
                    self._session_pitch_baseline = mean_pitch
                else:
                    self._session_pitch_baseline = (
                        0.8 * self._session_pitch_baseline + 0.2 * mean_pitch
                    )

            rms = librosa.feature.rms(y=y)
            mean_energy = float(np.mean(rms))

            self._energy_history.append(mean_energy)
            if self._session_energy_baseline is None:
                self._session_energy_baseline = mean_energy
            else:
                self._session_energy_baseline = float(
                    np.mean(self._energy_history[-10:])
                )

            speaking_rate = float(
                np.sum(voiced_flag) / max(len(voiced_flag), 1)
            )

            pitch_perturbation_ratio = 0.0
            if len(voiced_f0) > 1:
                diffs = np.abs(np.diff(voiced_f0))
                pitch_perturbation_ratio = float(
                    np.mean(diffs) / (np.mean(voiced_f0) + 1e-8)
                )

            valence_label, valence_score = self._estimate_valence(
                mean_pitch, pitch_std, mean_energy, speaking_rate
            )

            return {
                'mean_pitch_hz':            round(mean_pitch, 2),
                'pitch_std':                round(pitch_std, 2),
                'mean_energy':              round(mean_energy, 6),
                'speaking_rate':            round(speaking_rate, 4),
                'pitch_perturbation_ratio': round(pitch_perturbation_ratio, 6),
                'valence_estimate':         valence_label,
                'valence_score':            valence_score
            }

        except Exception as e:
            logger.error(
                "[ProsodyService] [extract] Failed on %s [%d-%d ms]: %s",
                audio_path, start_ms, end_ms, str(e)
            )
            return {}

    def _estimate_valence(
        self,
        pitch: float,
        pitch_std: float,
        energy: float,
        rate: float
    ) -> tuple:
        """
        Rule-based valence estimation using speaker-relative thresholds.

        Uses relative comparisons against session baselines rather than
        absolute Hz thresholds to avoid systematic gender bias.

        Returns:
            (valence_label, valence_score)
        """
        conditions = []

        if self._session_pitch_baseline and self._session_pitch_baseline > 0:
            conditions.append(pitch > self._session_pitch_baseline * 1.05)
        else:
            conditions.append(90.0 < pitch < 350.0)

        conditions.append(pitch_std > 20.0)

        if self._session_energy_baseline and self._session_energy_baseline > 0:
            conditions.append(energy > self._session_energy_baseline * 0.90)
        else:
            conditions.append(energy > 0.01)

        conditions.append(rate > 0.55)

        score = sum(conditions) / len(conditions)

        if score >= 0.67:
            return 'positive', round(score, 3)
        elif score <= 0.33:
            return 'negative', round(1.0 - score, 3)
        else:
            return 'neutral', round(0.5, 3)
