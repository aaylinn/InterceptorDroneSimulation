#!/usr/bin/env python3

import os
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ultralytics import YOLO


IMAGE_TOPIC = os.environ.get(
    "IMAGE_TOPIC",
    (
        "/world/interceptor_world/model/x500_mono_cam_0/"
        "link/camera_link/sensor/camera/image"
    ),
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/home/aylin/STAJ_INTERCEPTOR/weights/buoy_best.pt",
)

# --- YOLO parametreleri (env ile ayarlanabilir, hızlı deney için) ---
YOLO_CONF = float(os.environ.get("YOLO_CONF", 0.35))
YOLO_IOU = float(os.environ.get("YOLO_IOU", 0.45))

# Takip aktifken (last_box var) hizli/dusuk cozunurluklu tam-goruntu taramasi.
TRACK_IMGSZ = int(os.environ.get("TRACK_IMGSZ", 512))

# Takip kaybolduğunda / hic tespit yokken (arama modu) daha yuksek cozunurluk
# kullanilir. Uzak/kucuk nesnenin piksel detayini korumak icin bu onemli --
# 512'ye kucultmek zaten kucuk olan dubayi neredeyse yok ediyor.
SEARCH_IMGSZ = int(os.environ.get("SEARCH_IMGSZ", 800))

# ROI (zaten kirpilmis kucuk bolge) icin ayri, biraz daha yuksek imgsz.
ROI_IMGSZ = int(os.environ.get("ROI_IMGSZ", 640))

# Bu eşiğin üzerindeki confidence'lara "güven" -> HSV kırmızı-oran
# dogrulamasini tamamen atla. Boylece yuksek guvenli gercek tespitler,
# acidan kaynakli renk kaybi yuzunden reddedilmiyor.
YOLO_CONF_TRUST = float(os.environ.get("YOLO_CONF_TRUST", 0.55))

MIN_RED_CONTOUR_AREA = 70
MIN_RED_FILL_RATIO = 0.15
MIN_RED_RATIO_IN_YOLO_BOX = float(
    os.environ.get("MIN_RED_RATIO_IN_YOLO_BOX", 0.08)
)

# Kutu alani (px^2) bu esigin altindaysa (uzak/kucuk nesne), kirmizi-oran
# olcumu guvenilmez sayilir (birkac piksel gurultu orani tamamen degistirir).
# Bu durumda oran kontrolu atlanir, onun yerine ayri bir confidence tabani
# uygulanir (propeller gibi kucuk yanlis-pozitifleri yine de elemek icin).
SMALL_BOX_AREA_BYPASS = int(os.environ.get("SMALL_BOX_AREA_BYPASS", 900))
SMALL_BOX_MIN_CONF = float(os.environ.get("SMALL_BOX_MIN_CONF", 0.42))

# Ayni fiziksel dubanin farkli aci/isikta farkli siniflar (Red/Orange vb.)
# olarak tespit edilmesi normal -- ekranda kafa karistirmamasi icin sadelestir.
DISPLAY_AS_BUOY = os.environ.get("DISPLAY_AS_BUOY", "1") == "1"

# --- HSV-Direct: YOLO tamamen basarisiz oldugunda son care olarak
# saf kirmizi-renk tespitine guvenme ---
# Hedef "kirmizi dubayi vurmak" oldugu icin, sinif dogrulugu YOLO kadar
# kritik degil; uzak mesafede YOLO hic kutu uretmese bile HSV genelde
# adayi buluyor (log/ekran goruntulerinde HSV:ADAY neredeyse hep aktif).
# Bu yuzden YOLO (tam+ROI+fallback) hepsi basarisiz olursa, yeterince
# buyuk ve konumsal olarak tutarli bir HSV adayi dogrudan tespit olarak
# kabul edilir.
ENABLE_HSV_DIRECT = os.environ.get("ENABLE_HSV_DIRECT", "1") == "1"

# Temel MIN_RED_CONTOUR_AREA'dan (70) daha buyuk: dogrudan tespitte
# gurultuye karsi ekstra guvenlik payi.
HSV_DIRECT_MIN_AREA = int(os.environ.get("HSV_DIRECT_MIN_AREA", 130))

# Son bilinen kutuya gore izin verilen maksimum merkez kaymasi (px).
# Takip varken sahnedeki alakasiz bir kirmizi lekeye atlamayi engeller.
HSV_DIRECT_MAX_JUMP_PX = float(
    os.environ.get("HSV_DIRECT_MAX_JUMP_PX", 220.0)
)

# HSV-direct tespitlere atanan sentetik confidence (ekranda/loglama icin).
HSV_DIRECT_CONFIDENCE = 0.30

MAX_HOLD_FRAMES = int(os.environ.get("MAX_HOLD_FRAMES", 4))
SMOOTHING_ALPHA = 0.75

# last_box etrafinda fallback ROI denemesi icin buyutme katsayilari
FALLBACK_ROI_SCALE_W = 2.6
FALLBACK_ROI_SCALE_H = 2.2
FALLBACK_ROI_MAX_MISSED = 6  # last_box bu kadar karedir guncellenmediyse fallback'i deneme


class BuoyDetector(Node):
    def __init__(self):
        super().__init__("buoy_detector")

        cv2.setNumThreads(1)

        self.bridge = CvBridge()
        self.model = YOLO(MODEL_PATH)

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.subscription = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            image_qos,
        )

        # Kontrol kodunun okuyacagi taze hedef bilgisi.
        self.target_publisher = self.create_publisher(
            Float32MultiArray,
            "/buoy/target",
            10,
        )

        self.last_box: Optional[tuple[int, int, int, int]] = None
        self.missed_frames = 0

        self.total_frames = 0
        self.fresh_detections = 0
        self.roi_detections = 0
        self.full_detections = 0
        self.fallback_roi_detections = 0
        self.hsv_direct_detections = 0
        self.hold_frames = 0

        # Debug: hangi asamada tespit kaybediliyor
        self.rejected_low_ratio = 0
        self.rejected_small_area = 0
        self.rejected_small_low_conf = 0
        self.yolo_zero_box = 0

        self.previous_time = time.perf_counter()
        self.display_fps = 0.0

        self.get_logger().info(
            "Buoy detector baslatildi. Kademeli HSV dogrulama + fallback ROI etkin."
        )
        self.get_logger().info(f"Kamera topic: {IMAGE_TOPIC}")
        self.get_logger().info(
            f"YOLO_CONF={YOLO_CONF} YOLO_CONF_TRUST={YOLO_CONF_TRUST} "
            f"MIN_RED_RATIO_IN_YOLO_BOX={MIN_RED_RATIO_IN_YOLO_BOX}"
        )

    @staticmethod
    def create_red_mask(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_1 = cv2.inRange(
            hsv,
            np.array([0, 75, 45], dtype=np.uint8),
            np.array([14, 255, 255], dtype=np.uint8),
        )

        mask_2 = cv2.inRange(
            hsv,
            np.array([164, 75, 45], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )

        red_mask = cv2.bitwise_or(mask_1, mask_2)

        opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            opening_kernel,
            iterations=1,
        )

        closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        )
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            closing_kernel,
            iterations=2,
        )

        return red_mask

    @staticmethod
    def clamp_box(
        box: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box

        x1 = max(0, min(int(x1), frame_width - 1))
        y1 = max(0, min(int(y1), frame_height - 1))
        x2 = max(x1 + 1, min(int(x2), frame_width))
        y2 = max(y1 + 1, min(int(y2), frame_height))

        return x1, y1, x2, y2

    @staticmethod
    def red_ratio(
        red_mask: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> float:
        x1, y1, x2, y2 = box
        region = red_mask[y1:y2, x1:x2]

        if region.size == 0:
            return 0.0

        red_pixels = cv2.countNonZero(region)
        return float(red_pixels) / float(region.size)

    @staticmethod
    def required_red_ratio(confidence: float) -> float:
        """
        Confidence yukseldikce gereken kirmizi-oran esigini dusurur.
        confidence >= YOLO_CONF_TRUST ise 0.0 (tamamen atla, modele guven).
        confidence == YOLO_CONF ise tam MIN_RED_RATIO_IN_YOLO_BOX uygulanir.
        Arada dogrusal gecis yapilir. Bu, pervane gibi dusuk-confidence
        false-positive'leri hala HSV ile elerken, acidan kaynakli renk
        kaybi yasayan yuksek-confidence gercek tespitlerin kurtulmasini saglar.
        """
        if confidence >= YOLO_CONF_TRUST:
            return 0.0

        span = max(1e-6, YOLO_CONF_TRUST - YOLO_CONF)
        progress = (confidence - YOLO_CONF) / span
        progress = max(0.0, min(1.0, progress))

        return MIN_RED_RATIO_IN_YOLO_BOX * (1.0 - progress)

    @staticmethod
    def display_class_name(class_name: str) -> str:
        if DISPLAY_AS_BUOY:
            return f"Buoy({class_name})"

        return class_name

    def class_name(self, class_id: int) -> str:
        names = self.model.names

        if isinstance(names, dict):
            return str(names.get(class_id, class_id))

        if 0 <= class_id < len(names):
            return str(names[class_id])

        return str(class_id)

    def find_red_candidate(
        self,
        red_mask: np.ndarray,
    ) -> Optional[tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        previous_center = None

        if self.last_box is not None:
            x1, y1, x2, y2 = self.last_box
            previous_center = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            )

        best_box = None
        best_score = -1.0

        for contour in contours:
            contour_area = cv2.contourArea(contour)

            if contour_area < MIN_RED_CONTOUR_AREA:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            if width < 4 or height < 4:
                continue

            rectangle_area = width * height
            fill_ratio = contour_area / float(rectangle_area)

            if fill_ratio < MIN_RED_FILL_RATIO:
                continue

            aspect_ratio = height / float(width)

            if aspect_ratio < 0.25 or aspect_ratio > 6.0:
                continue

            center_x = x + width / 2.0
            center_y = y + height / 2.0

            score = contour_area

            if previous_center is not None:
                distance = np.hypot(
                    center_x - previous_center[0],
                    center_y - previous_center[1],
                )
                score = score / (1.0 + distance / 250.0)

            if score > best_score:
                best_score = score
                best_box = (
                    x,
                    y,
                    x + width,
                    y + height,
                )

        return best_box

    @staticmethod
    def create_fixed_roi(
        frame: np.ndarray,
        candidate_box: tuple[int, int, int, int],
        scale_w: float = 2.4,
        scale_h: float = 1.8,
    ):
        """
        Hedef merkezde veya goruntu kenarinda olsa da ROI boyutunu
        mumkun oldugunca sabit tutar.
        """
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = candidate_box

        candidate_width = max(1, x2 - x1)
        candidate_height = max(1, y2 - y1)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        roi_width = max(int(candidate_width * scale_w), 140)
        roi_height = max(int(candidate_height * scale_h), 140)

        roi_width = min(roi_width, frame_width)
        roi_height = min(roi_height, frame_height)

        roi_x1 = center_x - roi_width // 2
        roi_y1 = center_y - roi_height // 2
        roi_x2 = roi_x1 + roi_width
        roi_y2 = roi_y1 + roi_height

        if roi_x1 < 0:
            roi_x2 -= roi_x1
            roi_x1 = 0

        if roi_y1 < 0:
            roi_y2 -= roi_y1
            roi_y1 = 0

        if roi_x2 > frame_width:
            shift = roi_x2 - frame_width
            roi_x1 -= shift
            roi_x2 = frame_width

        if roi_y2 > frame_height:
            shift = roi_y2 - frame_height
            roi_y1 -= shift
            roi_y2 = frame_height

        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(frame_width, roi_x2)
        roi_y2 = min(frame_height, roi_y2)

        if roi_x2 - roi_x1 < 30 or roi_y2 - roi_y1 < 30:
            return None

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        return roi, roi_x1, roi_y1

    def extract_best_detection(
        self,
        result,
        full_red_mask: np.ndarray,
        full_frame_shape: tuple[int, int, int],
        offset_x: int = 0,
        offset_y: int = 0,
    ):
        if result.boxes is None or len(result.boxes) == 0:
            self.yolo_zero_box += 1
            return None

        frame_height, frame_width = full_frame_shape[:2]

        best_detection = None
        best_score = -1.0

        for box_data in result.boxes:
            confidence = float(box_data.conf[0].cpu().item())
            class_id = int(box_data.cls[0].cpu().item())

            x1, y1, x2, y2 = (
                box_data.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            full_box = self.clamp_box(
                (
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                ),
                frame_width,
                frame_height,
            )

            bx1, by1, bx2, by2 = full_box
            box_area = (bx2 - bx1) * (by2 - by1)

            if box_area < 100:
                self.rejected_small_area += 1
                continue

            if box_area < SMALL_BOX_AREA_BYPASS:
                # Kutu cok kucuk (uzak nesne): kirmizi-oran olcumu birkac
                # pikselden ibaret olabilir, gurultuye asiri duyarli.
                # HSV oranina guvenmek yerine ayri, biraz daha siki bir
                # confidence tabani uygulanir.
                if confidence < SMALL_BOX_MIN_CONF:
                    self.rejected_small_low_conf += 1
                    continue

                box_red_ratio = self.red_ratio(
                    full_red_mask,
                    full_box,
                )
            else:
                box_red_ratio = self.red_ratio(
                    full_red_mask,
                    full_box,
                )

                required_ratio = self.required_red_ratio(confidence)

                if box_red_ratio < required_ratio:
                    self.rejected_low_ratio += 1
                    continue

            # confidence + kirmizi-oran karisik skor; trust bolgesinde
            # kirmizi-oran katkisini kucult ki gereksiz yere ROI/tam
            # goruntu arasinda farkli kutular secilmesin.
            score = confidence + 0.12 * box_red_ratio

            if score > best_score:
                best_score = score
                best_detection = {
                    "box": full_box,
                    "confidence": confidence,
                    "class_name": self.class_name(class_id),
                }

        return best_detection

    def try_hsv_direct_detection(
        self,
        candidate_box: Optional[tuple[int, int, int, int]],
    ):
        """
        YOLO (tam + ROI + fallback) hicbir kutu uretemediginde son care.
        Yeterince buyuk ve (varsa) son bilinen konuma yakin bir HSV
        adayini dogrudan tespit olarak kabul eder. Sinif bilgisi yok --
        amac zaten "kirmizi nesneyi bul", ince siniflandirma degil.
        """
        if not ENABLE_HSV_DIRECT or candidate_box is None:
            return None

        x1, y1, x2, y2 = candidate_box
        area = max(0, x2 - x1) * max(0, y2 - y1)

        if area < HSV_DIRECT_MIN_AREA:
            return None

        if self.last_box is not None:
            lx1, ly1, lx2, ly2 = self.last_box
            last_center = ((lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0)
            new_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            jump = np.hypot(
                new_center[0] - last_center[0],
                new_center[1] - last_center[1],
            )

            if jump > HSV_DIRECT_MAX_JUMP_PX:
                return None

        return {
            "box": candidate_box,
            "confidence": HSV_DIRECT_CONFIDENCE,
            "class_name": "Red",
        }

    def run_yolo(
        self,
        image: np.ndarray,
        full_red_mask: np.ndarray,
        full_frame_shape: tuple[int, int, int],
        offset_x: int = 0,
        offset_y: int = 0,
        imgsz: int = TRACK_IMGSZ,
    ):
        results = self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            max_det=1,
            verbose=False,
        )

        return self.extract_best_detection(
            results[0],
            full_red_mask,
            full_frame_shape,
            offset_x,
            offset_y,
        )

    def smooth_box(
        self,
        new_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        if self.last_box is None:
            return new_box

        values = []

        for old_value, new_value in zip(
            self.last_box,
            new_box,
        ):
            value = (
                SMOOTHING_ALPHA * new_value
                + (1.0 - SMOOTHING_ALPHA) * old_value
            )
            values.append(int(value))

        return tuple(values)

    def update_fps(self):
        current_time = time.perf_counter()
        elapsed = current_time - self.previous_time
        self.previous_time = current_time

        if elapsed <= 0:
            return

        instant_fps = 1.0 / elapsed

        if self.display_fps == 0.0:
            self.display_fps = instant_fps
        else:
            self.display_fps = (
                0.85 * self.display_fps
                + 0.15 * instant_fps
            )

    @staticmethod
    def draw_box(
        frame: np.ndarray,
        box: tuple[int, int, int, int],
        text: str,
        color: tuple[int, int, int],
    ):
        x1, y1, x2, y2 = box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            frame,
            text,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    def publish_target(
        self,
        frame_shape,
        detection,
        detection_source,
    ):
        """
        /buoy/target veri sirasi:

        [0] visible        : 1=taze tespit, 0=tespit yok/HOLD
        [1] center_x_norm  : 0.0 sol, 0.5 merkez, 1.0 sag
        [2] center_y_norm  : 0.0 ust, 0.5 merkez, 1.0 alt
        [3] width_ratio    : kutu genisligi / goruntu genisligi
        [4] height_ratio   : kutu yuksekligi / goruntu yuksekligi
        [5] area_ratio     : kutu alani / goruntu alani
        [6] confidence     : YOLO confidence veya HSV icin 0.30
        [7] source_code    : 1=YOLO, 2=ROI, 3=Fallback, 4=HSV
        """
        message = Float32MultiArray()

        if detection is None or detection_source is None:
            message.data = [
                0.0,
                0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            self.target_publisher.publish(message)
            return

        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = detection["box"]

        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        source_codes = {
            "YOLO": 1.0,
            "YOLO-ROI": 2.0,
            "YOLO-FALLBACK": 3.0,
            "HSV": 4.0,
        }

        message.data = [
            1.0,
            float(center_x / frame_width),
            float(center_y / frame_height),
            float(box_width / frame_width),
            float(box_height / frame_height),
            float(
                (box_width * box_height)
                / (frame_width * frame_height)
            ),
            float(detection["confidence"]),
            source_codes.get(detection_source, 0.0),
        ]

        self.target_publisher.publish(message)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            self.total_frames += 1
            self.update_fps()

            red_mask = self.create_red_mask(frame)
            annotated_frame = frame.copy()

            candidate_box = self.find_red_candidate(
                red_mask,
            )

            # Aktif takip yoksa (nesne az once kayboldu / hic bulunamadi)
            # arama modunda daha yuksek cozunurluk kullanilir; bu, uzak/kucuk
            # nesnenin piksel detayinin 512'ye kucultulurken yok olmasini onler.
            full_frame_imgsz = (
                TRACK_IMGSZ if self.last_box is not None else SEARCH_IMGSZ
            )

            # Once tam goruntude yalnizca bir YOLO calistirilir.
            detection = self.run_yolo(
                frame,
                red_mask,
                frame.shape,
                imgsz=full_frame_imgsz,
            )

            detection_source = None

            if detection is not None:
                detection_source = "YOLO"
                self.full_detections += 1

            # Tam goruntu kacirirsa sabit boyutlu ROI yedek olarak denenir.
            elif candidate_box is not None:
                roi_data = self.create_fixed_roi(
                    frame,
                    candidate_box,
                )

                if roi_data is not None:
                    roi, offset_x, offset_y = roi_data

                    detection = self.run_yolo(
                        roi,
                        red_mask,
                        frame.shape,
                        offset_x,
                        offset_y,
                        imgsz=ROI_IMGSZ,
                    )

                    if detection is not None:
                        detection_source = "YOLO-ROI"
                        self.roi_detections += 1

            # HSV de aday bulamadiysa (renk kaybi/aci degisimi) ama yakin
            # zamanda gecerli bir last_box varsa, onun etrafinda genisletilmis
            # bir ROI ile son bir deneme yapilir. Bu, HSV'nin de basarisiz
            # oldugu acilarda kurtarma sansi verir.
            if (
                detection is None
                and candidate_box is None
                and self.last_box is not None
                and self.missed_frames <= FALLBACK_ROI_MAX_MISSED
            ):
                roi_data = self.create_fixed_roi(
                    frame,
                    self.last_box,
                    scale_w=FALLBACK_ROI_SCALE_W,
                    scale_h=FALLBACK_ROI_SCALE_H,
                )

                if roi_data is not None:
                    roi, offset_x, offset_y = roi_data

                    detection = self.run_yolo(
                        roi,
                        red_mask,
                        frame.shape,
                        offset_x,
                        offset_y,
                        imgsz=ROI_IMGSZ,
                    )

                    if detection is not None:
                        detection_source = "YOLO-FALLBACK"
                        self.fallback_roi_detections += 1

            # Son care: YOLO (tam+ROI+fallback) hicbir kutu uretemedi.
            # Yeterince buyuk ve konumsal olarak tutarli bir HSV adayi
            # varsa, onu dogrudan tespit olarak kullan. Amac zaten kirmizi
            # nesneyi bulmak oldugu icin, YOLO'nun tamamen sessiz kaldigi
            # uzak mesafelerde rehberligi HSV ile ayakta tutar.
            if detection is None:
                detection = self.try_hsv_direct_detection(candidate_box)

                if detection is not None:
                    detection_source = "HSV"
                    self.hsv_direct_detections += 1

            display_box = None
            display_text = None
            display_color = (0, 255, 0)

            if detection is not None:
                display_box = self.smooth_box(
                    detection["box"],
                )

                self.last_box = display_box
                self.missed_frames = 0
                self.fresh_detections += 1

                display_text = (
                    f'{self.display_class_name(detection["class_name"])} '
                    f'{detection["confidence"]:.2f} '
                    f'{detection_source}'
                )

                if detection_source == "YOLO-ROI":
                    display_color = (255, 255, 0)
                elif detection_source == "YOLO-FALLBACK":
                    display_color = (255, 128, 0)
                elif detection_source == "HSV":
                    display_color = (255, 0, 255)

            else:
                self.missed_frames += 1

                if (
                    self.last_box is not None
                    and self.missed_frames <= MAX_HOLD_FRAMES
                ):
                    display_box = self.last_box
                    display_text = (
                        f"HOLD "
                        f"{self.missed_frames}/{MAX_HOLD_FRAMES}"
                    )
                    display_color = (0, 255, 255)
                    self.hold_frames += 1
                else:
                    self.last_box = None

            # HOLD gorsellestirilse bile detection None kalir.
            # Dolayisiyla kontrol sistemi eski kutuyla ileri gitmez.
            self.publish_target(
                frame.shape,
                detection,
                detection_source,
            )

            if display_box is not None:
                self.draw_box(
                    annotated_frame,
                    display_box,
                    display_text,
                    display_color,
                )

            detection_rate = (
                100.0
                * self.fresh_detections
                / max(1, self.total_frames)
            )

            hsv_text = (
                "HSV:ADAY"
                if candidate_box is not None
                else "HSV:YOK"
            )

            status_text = (
                f"FPS:{self.display_fps:.1f}  "
                f"Fresh:{self.fresh_detections}/{self.total_frames} "
                f"(%{detection_rate:.1f})  "
                f"ROI:{self.roi_detections}  "
                f"Full:{self.full_detections}  "
                f"FB:{self.fallback_roi_detections}  "
                f"HSVd:{self.hsv_direct_detections}  "
                f"Hold:{self.hold_frames}  "
                f"{hsv_text}"
            )

            cv2.putText(
                annotated_frame,
                status_text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Buoy YOLO Detection",
                annotated_frame,
            )
            cv2.waitKey(1)

            if self.total_frames % 30 == 0:
                self.get_logger().info(
                    f"Taze tespit: %{detection_rate:.1f} | "
                    f"ROI: {self.roi_detections} | "
                    f"Tam goruntu: {self.full_detections} | "
                    f"Fallback: {self.fallback_roi_detections} | "
                    f"HSV-direct: {self.hsv_direct_detections} | "
                    f"Hold: {self.hold_frames} | "
                    f"Red: yolo_zero_box={self.yolo_zero_box} "
                    f"rejected_low_ratio={self.rejected_low_ratio} "
                    f"rejected_small_area={self.rejected_small_area} "
                    f"rejected_small_low_conf={self.rejected_small_low_conf}"
                )

        except Exception as error:
            self.get_logger().error(
                f"Goruntu isleme hatasi: {error}"
            )


def main():
    rclpy.init()
    node = BuoyDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
