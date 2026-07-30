"""
Gorev 2 karsilastirma logu.

health_status==1 anlarinda sunucudan gelen GERCEK (GT) translation ile,
health_status==0 anlarinda bizim gonderdigimiz TAHMIN translation'ini ayni
CSV'ye satir satir yazar. Oturum sonunda kullanicinin elindeki gercek
verilerle karsilastirilip ne kadar saptigimiz gorulebilsin diyedir.
"""
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

FIELDNAMES = ["timestamp", "frame_image_url", "health_status", "source", "x", "y", "z"]


class TranslationComparisonLogger:
    def __init__(self, log_folder="./_logs/"):
        Path(log_folder).mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("translation_karsilastirma_%Y_%m_%d__%H_%M_%S_%f.csv")
        self.path = os.path.join(log_folder, filename)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()
        logging.info(f"TranslationComparisonLogger: writing to {self.path}")

    def log_server_gt(self, frame_image_url, x, y, z):
        self._write(frame_image_url, health_status="1", source="server_gt", x=x, y=y, z=z)

    def log_our_prediction(self, frame_image_url, x, y, z):
        self._write(frame_image_url, health_status="0", source="pred_ours", x=x, y=y, z=z)

    def _write(self, frame_image_url, health_status, source, x, y, z):
        self._writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "frame_image_url": frame_image_url,
            "health_status": health_status,
            "source": source,
            "x": x, "y": y, "z": z,
        })
        self._file.flush()

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass
