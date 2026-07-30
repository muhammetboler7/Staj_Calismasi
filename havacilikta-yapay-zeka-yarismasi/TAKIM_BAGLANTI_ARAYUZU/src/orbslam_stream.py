"""
Canli ORB-SLAM3 kosumu icin Python sarmalayicisi (Gorev 2).

gps_denied_localization/tools/ORB_SLAM3/Examples/Monocular/mono_tuyz_stream
surecini bir kere baslatir ve oturum boyunca ACIK tutar (ORB-SLAM3'un
haritasi/izleme durumu surec boyunca birikir). Her yeni kare indirildikce
track() ile o karenin dosya yolu gonderilir, karsiliginda o karenin online
kamera pozu (Twc, dunya<-kamera) okunur.

Protokol iki ayri kanal kullanir (bkz. mono_tuyz_stream.cc basindaki not):
  - stdin  : "<frame_id> <mutlak_goruntu_yolu>"  (komut)
  - proto pipe (ayri bir fd):
      "<frame_id> OK tx ty tz qx qy qz qw med_depth n_tracked"
      | "<frame_id> FAIL" | "<frame_id> ERR ..."
    med_depth: takip edilen harita noktalarinin kamera-cercevesi derinlik
    medyani (Z tahmini icin; nokta yoksa -1), n_tracked: nokta sayisi.
ORB-SLAM3 kutuphanesinin kendi durum mesajlarini bastirdigi stdout/stderr
bilerek yoksayilir (bir log dosyasina yonlendirilir) - protokolle karismasin.
"""
import logging
import os
import select
import subprocess
import time
from pathlib import Path

# TAKIM_BAGLANTI_ARAYUZU/src/orbslam_stream.py -> ... -> Muhammet_Boler
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ORBSLAM_ROOT = _REPO_ROOT / "gps_denied_localization" / "tools" / "ORB_SLAM3"

DEFAULT_BINARY = _ORBSLAM_ROOT / "Examples" / "Monocular" / "mono_tuyz_stream"
DEFAULT_VOCAB = _ORBSLAM_ROOT / "Vocabulary" / "ORBvoc.txt"
DEFAULT_SETTINGS = _REPO_ROOT / "gps_denied_localization" / "config" / "thyz2026.yaml"

STARTUP_TIMEOUT_S = 180.0  # ORBvoc.txt yuklemesi (metin format) yavas olabilir


class OrbSlamStreamError(RuntimeError):
    pass


class OrbSlamStream:
    """mono_tuyz_stream surecini yonetir. track() her karede cagrilir."""

    def __init__(self, binary_path=None, vocab_path=None, settings_path=None, stderr_log_path=None):
        self.binary_path = Path(binary_path) if binary_path else DEFAULT_BINARY
        self.vocab_path = Path(vocab_path) if vocab_path else DEFAULT_VOCAB
        self.settings_path = Path(settings_path) if settings_path else DEFAULT_SETTINGS

        for p, label in ((self.binary_path, "binary"), (self.vocab_path, "vocab"), (self.settings_path, "settings")):
            if not p.exists():
                raise OrbSlamStreamError(f"ORB-SLAM3 stream {label} not found: {p}")

        read_fd, write_fd = os.pipe()
        os.set_inheritable(write_fd, True)

        stderr_dest = open(stderr_log_path, "w") if stderr_log_path else subprocess.DEVNULL
        stdout_dest = open(str(stderr_log_path) + ".stdout", "w") if stderr_log_path else subprocess.DEVNULL

        try:
            self._proc = subprocess.Popen(
                [str(self.binary_path), str(self.vocab_path), str(self.settings_path), f"--proto-fd={write_fd}"],
                stdin=subprocess.PIPE,
                stdout=stdout_dest,
                stderr=stderr_dest,
                text=True,
                bufsize=1,
                pass_fds=(write_fd,),
            )
        finally:
            os.close(write_fd)

        self._proto = os.fdopen(read_fd, "r")
        self._closed = False
        self._counter = 0
        self._wait_ready()
        logging.info("OrbSlamStream: ORB-SLAM3 streaming process ready (pid=%s).", self._proc.pid)

    def _wait_ready(self):
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while True:
            if self._proc.poll() is not None:
                raise OrbSlamStreamError(
                    f"ORB-SLAM3 stream process exited during startup (code={self._proc.returncode}).")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OrbSlamStreamError("ORB-SLAM3 stream process did not become READY in time.")
            ready, _, _ = select.select([self._proto], [], [], remaining)
            if not ready:
                continue
            line = self._proto.readline()
            if line.strip() == "READY":
                return

    def track(self, image_path):
        """Bir kareyi ORB-SLAM3'e besler.

        Returns: (x, y, z, qx, qy, qz, qw, med_depth, n_tracked) basarili
        takipte (med_depth: takip edilen noktalarin kamera-cercevesi derinlik
        medyani, nokta yoksa -1.0; n_tracked: nokta sayisi), None ise
        (tracking kaybedildi / goruntu okunamadi).
        Raises OrbSlamStreamError: surec beklenmedik sekilde oldu ise.
        """
        if self._closed:
            raise OrbSlamStreamError("track() called after close().")

        frame_id = str(self._counter)
        self._counter += 1

        try:
            self._proc.stdin.write(f"{frame_id} {image_path}\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as e:
            raise OrbSlamStreamError(f"ORB-SLAM3 stream process is not accepting input: {e}") from e

        line = self._proto.readline()
        if not line:
            raise OrbSlamStreamError(
                f"ORB-SLAM3 stream process closed the protocol channel unexpectedly "
                f"(exit code={self._proc.poll()}).")

        parts = line.split()
        if not parts or parts[0] != frame_id:
            raise OrbSlamStreamError(
                f"ORB-SLAM3 stream protocol desync: expected frame_id={frame_id}, got line={line!r}")

        status = parts[1]
        if status == "OK":
            x, y, z, qx, qy, qz, qw, med_depth = (float(v) for v in parts[2:10])
            n_tracked = int(parts[10])
            return (x, y, z, qx, qy, qz, qw, med_depth, n_tracked)
        if status == "FAIL":
            return None
        if status == "ERR":
            logging.warning("OrbSlamStream: frame %s error: %s", frame_id, " ".join(parts[2:]))
            return None
        raise OrbSlamStreamError(f"ORB-SLAM3 stream protocol: unknown status line: {line!r}")

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.stdin.write("QUIT\n")
            self._proc.stdin.flush()
        except Exception:
            pass
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=15)
        except Exception:
            self._proc.kill()
        try:
            self._proto.close()
        except Exception:
            pass
        logging.info("OrbSlamStream: closed.")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
