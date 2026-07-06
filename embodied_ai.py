import time
from typing import List, Optional, Tuple


import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO


from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.g1.arm.g1_arm_action_client import action_map
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient




class EmbodiedAI:
   """
   Abstração unificada das ações de braço (G1ArmActionClient),
   locomoção (LocoClient), percepção visual (YOLO) e profundidade
   (RealSense RGB-D) do Unitree G1 EDU.


   Uso:
       robot = EmbodiedAI("eth0")
       robot.clap()
       robot.move_forward(0.3)


       deteccoes = robot.detectar()
       # [{"classe": "person", "confianca": 0.91, "bbox": (x1,y1,x2,y2)}, ...]


       distancia = robot.distancia_objeto("person")
       # 1.42  (metros)
   """


   _initialized = False  # garante que ChannelFactoryInitialize seja chamado uma única vez


   def __init__(
       self,
       network_interface: str,
       timeout: float = 10.0,
       modelo_yolo: str = "yolo26m.pt",
       confianca_min: float = 0.5,
       janela_depth: int = 10,
   ):
       """
       Inicializa o canal DDS, clientes de braço/locomoção,
       câmera RealSense e modelo YOLO.


       Args:
           network_interface: Interface de rede DDS (ex: "eth0").
           timeout:           Timeout dos clientes Unitree, em segundos.
           modelo_yolo:       Arquivo .pt do modelo YOLO a carregar.
           confianca_min:     Limiar mínimo de confiança para detecções.
           janela_depth:      Meia-janela (px) para mediana de profundidade.
       """
       if not EmbodiedAI._initialized:
           ChannelFactoryInitialize(0, network_interface)
           EmbodiedAI._initialized = True


       self._arm = G1ArmActionClient()
       self._arm.SetTimeout(timeout)
       self._arm.Init()


       self._loco = LocoClient()
       self._loco.SetTimeout(timeout)
       self._loco.Init()


       self._confianca_min = confianca_min
       self._janela_depth  = janela_depth


       # ── YOLO: tenta GPU, cai para CPU se não disponível ──────────────
       if torch.cuda.is_available():
           self._device = "cuda"
           print(f"  [EmbodiedAI] YOLO → GPU ({torch.cuda.get_device_name(0)})")
       else:
           self._device = "cpu"
           print("  [EmbodiedAI] CUDA indisponível — YOLO → CPU")


       self._yolo = YOLO(modelo_yolo)
       self._yolo.to(self._device)


       # ── RealSense ─────────────────────────────────────────────────────
       self._pipeline, self._align = self._iniciar_camera()


   # ------------------------------------------------------------------
   # Visão — câmera e YOLO (métodos internos)
   # ------------------------------------------------------------------


   @staticmethod
   def _iniciar_camera() -> Tuple[rs.pipeline, rs.align]:
       """Inicia o pipeline RealSense alinhando profundidade ao RGB."""
       pipeline = rs.pipeline()
       config   = rs.config()
       config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
       config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
       try:
           pipeline.start(config)
       except RuntimeError as e:
           if "errno=16" in str(e) or "busy" in str(e).lower():
               print("  [EmbodiedAI] Câmera ocupada — tentando novamente em 3s...")
               time.sleep(3)
               pipeline.start(config)
           else:
               raise
       print("  [EmbodiedAI] Câmera RealSense iniciada")
       return pipeline, rs.align(rs.stream.color)


   def _capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[rs.depth_frame]]:
       """Captures and aligns a (BGR image, depth_frame) pair. Returns (None, None) if invalid."""
       frames      = self._pipeline.wait_for_frames()
       aligned     = self._align.process(frames)
       color_frame = aligned.get_color_frame()
       depth_frame = aligned.get_depth_frame()
       if not color_frame or not depth_frame:
           return None, None
       return np.asanyarray(color_frame.get_data()), depth_frame


   def _median_depth(
       self,
       depth_frame: rs.depth_frame,
       cx: int,
       cy: int,
   ) -> float:
       """
       Computes the median distance (metres) inside a window centred at (cx, cy),
       ignoring zero-depth pixels. Returns 0.0 if the entire region is invalid.
       """
       j = self._janela_depth
       x1, x2 = max(0, cx - j), min(640, cx + j)
       y1, y2 = max(0, cy - j), min(480, cy + j)
       arr = np.asanyarray(depth_frame.get_data()).astype(float)
       reg = arr[y1:y2, x1:x2]
       reg[reg == 0] = np.nan
       if np.all(np.isnan(reg)):
           return 0.0
       return float(np.nanmedian(reg)) * depth_frame.get_units()


   # ------------------------------------------------------------------
   # ------------------------------------------------------------------
   # Vision — public API
   # ------------------------------------------------------------------


   def detect(self) -> List[dict]:
       """
       Captures one frame and runs YOLO inference.


       Returns a list of dicts, one per detection above the confidence threshold:
           {
               "class"      : str,            # class name (e.g. "person")
               "confidence" : float,          # 0.0 – 1.0
               "bbox"       : (x1,y1,x2,y2), # pixels, int
               "center"     : (cx, cy)        # pixels, int
           }
       Returns an empty list if no valid frame is available.
       """
       img, _ = self._capture_frame()
       if img is None:
           return []


       results      = self._yolo(img, verbose=False)
       detections   = []
       for result in results:
           for box in result.boxes:
               conf = float(box.conf[0])
               if conf < self._confianca_min:
                   continue
               x1, y1, x2, y2 = map(int, box.xyxy[0])
               detections.append({
                   "class"      : self._yolo.names[int(box.cls[0])],
                   "confidence" : round(conf, 3),
                   "bbox"       : (x1, y1, x2, y2),
                   "center"     : ((x1 + x2) // 2, (y1 + y2) // 2),
               })
       return detections


   def object_distance(self, class_name: Optional[str] = None) -> Optional[float]:
       """
       Returns the distance (metres) to the closest detected object.


       Args:
           class_name: If provided, filters detections to this class only
                       (e.g. "person"). If None, considers all classes.


       Returns:
           Distance in metres to the closest valid object, or None if
           no valid detection is found.


       Example:
           dist = robot.object_distance("person")  # → 1.42
           dist = robot.object_distance()           # → closest object of any class
       """
       img, depth_frame = self._capture_frame()
       if img is None or depth_frame is None:
           return None


       results   = self._yolo(img, verbose=False)
       distances = []


       for result in results:
           for box in result.boxes:
               conf = float(box.conf[0])
               if conf < self._confianca_min:
                   continue
               name = self._yolo.names[int(box.cls[0])]
               if class_name is not None and name != class_name:
                   continue
               x1, y1, x2, y2 = map(int, box.xyxy[0])
               cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
               d = self._median_depth(depth_frame, cx, cy)
               if d > 0.0:
                   distances.append(d)


       return round(min(distances), 3) if distances else None


   def detect_with_distance(
       self,
       class_name: Optional[str] = None,
   ) -> List[dict]:
       """
       Captures a single frame, runs YOLO inference, and enriches each
       detection with the depth-camera distance — all in one shot.


       Args:
           class_name: If provided, returns only detections of this class.
                       If None, returns all classes above the confidence threshold.


       Returns:
           List of dicts, one per detection:
               {
                   "class"      : str,            # class name (e.g. "person")
                   "confidence" : float,          # 0.0 – 1.0
                   "bbox"       : (x1,y1,x2,y2), # pixels, int
                   "center"     : (cx, cy),       # pixels, int
                   "distance_m" : float           # metres; 0.0 if depth unavailable
               }
           Returns an empty list if no valid frame is available.


       Example:
           detections = robot.detect_with_distance("person")
           for d in detections:
               print(d["class"], d["confidence"], d["distance_m"])
       """
       img, depth_frame = self._capture_frame()
       if img is None or depth_frame is None:
           return []


       results    = self._yolo(img, verbose=False)
       detections = []


       for result in results:
           for box in result.boxes:
               conf = float(box.conf[0])
               if conf < self._confianca_min:
                   continue
               name = self._yolo.names[int(box.cls[0])]
               if class_name is not None and name != class_name:
                   continue
               x1, y1, x2, y2 = map(int, box.xyxy[0])
               cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
               dist = self._median_depth(depth_frame, cx, cy)
               detections.append({
                   "class"      : name,
                   "confidence" : round(conf, 3),
                   "bbox"       : (x1, y1, x2, y2),
                   "center"     : (cx, cy),
                   "distance_m" : round(dist, 3),
               })


       return detections


   def close_camera(self) -> None:
       """Stops the RealSense pipeline."""
       self._pipeline.stop()
       print("  [EmbodiedAI] Camera closed")


   # ------------------------------------------------------------------
   # Método auxiliar interno — braço
   # ------------------------------------------------------------------


   def _execute(self, action_name: str) -> None:
       """Executa uma ação de braço pelo nome exato registrado em action_map."""
       self._arm.ExecuteAction(action_map.get(action_name))


   # ------------------------------------------------------------------
   # Ações disponíveis
   # ------------------------------------------------------------------


   def release_arm(self) -> None:
       """Libera os braços para a posição neutra."""
       self._execute("release arm")


   def shake_hand(self) -> None:
       """Cumprimenta com aperto de mão e retorna à posição neutra."""
       self._execute("shake hand")
       time.sleep(2)
       self.release_arm()


   def high_five(self) -> None:
       """Realiza gesto de 'high five' e retorna à posição neutra."""
       self._execute("high five")
       time.sleep(2)
       self.release_arm()


   def hug(self) -> None:
       """Realiza gesto de abraço e retorna à posição neutra."""
       self._execute("hug")
       time.sleep(2)
       self.release_arm()


   def high_wave(self) -> None:
       """Acena com o braço levantado."""
       self._execute("high wave")


   def clap(self) -> None:
       """Realiza gesto de palmas."""
       self._execute("clap")


   def face_wave(self) -> None:
       """Acena na altura do rosto."""
       self._execute("face wave")


   def left_kiss(self) -> None:
       """Envia beijo com a mão esquerda."""
       self._execute("left kiss")


   def heart(self) -> None:
       """Forma coração com as duas mãos e retorna à posição neutra."""
       self._execute("heart")
       time.sleep(2)
       self.release_arm()


   def right_heart(self) -> None:
       """Forma coração com a mão direita e retorna à posição neutra."""
       self._execute("right heart")
       time.sleep(2)
       self.release_arm()


   def hands_up(self) -> None:
       """Levanta as duas mãos e retorna à posição neutra."""
       self._execute("hands up")
       time.sleep(2)
       self.release_arm()


   def x_ray(self) -> None:
       """Realiza pose de raio-X e retorna à posição neutra."""
       self._execute("x-ray")
       time.sleep(2)
       self.release_arm()


   def right_hand_up(self) -> None:
       """Levanta a mão direita e retorna à posição neutra."""
       self._execute("right hand up")
       time.sleep(2)
       self.release_arm()


   def reject(self) -> None:
       """Realiza gesto de rejeição e retorna à posição neutra."""
       self._execute("reject")
       time.sleep(2)
       self.release_arm()


   def right_kiss(self) -> None:
       """Envia beijo com a mão direita."""
       self._execute("right kiss")


   def two_hand_kiss(self) -> None:
       """Envia beijo com as duas mãos."""
       self._execute("two-hand kiss")


   # ------------------------------------------------------------------
   # Locomoção — postura
   # ------------------------------------------------------------------


   def damp(self) -> None:
       """Ativa modo amortecido (baixa rigidez nas juntas)."""
       self._loco.Damp()


   def zero_torque(self) -> None:
       """Zera o torque em todas as juntas."""
       self._loco.ZeroTorque()


   def low_stand(self) -> None:
       """Postura em pé baixa."""
       self._loco.LowStand()


   def high_stand(self) -> None:
       """Postura em pé alta."""
       self._loco.HighStand()


   def squat_to_standup(self) -> None:
       """Transição de agachado para em pé (requer damp antes)."""
       self._loco.Damp()
       time.sleep(0.5)
       self._loco.Squat2StandUp()


   def standup_to_squat(self) -> None:
       """Transição de em pé para agachado."""
       self._loco.StandUp2Squat()


   def lie_to_standup(self) -> None:
       """
       Levanta o robô a partir da posição deitado.


       ATENÇÃO: o robô deve estar de barriga para cima, sobre
       superfície dura, plana e antiderrapante.
       """
       self._loco.Damp()
       time.sleep(0.5)
       self._loco.Lie2StandUp()


   # ------------------------------------------------------------------
   # Locomoção — movimento
   # ------------------------------------------------------------------


   def move(self, vx: float, vy: float, vyaw: float) -> None:
       """
       Envia comando de velocidade diretamente.


       Args:
           vx:   Velocidade linear frontal  (m/s).
           vy:   Velocidade linear lateral  (m/s).
           vyaw: Velocidade angular de giro (rad/s).
       """
       self._loco.Move(vx, vy, vyaw)


   def move_forward(self, speed: float = 0.3) -> None:
       """Avança para frente. speed em m/s."""
       self._loco.Move(speed, 0, 0)


   def move_lateral(self, speed: float = 0.3) -> None:
       """Move lateralmente (positivo = esquerda). speed em m/s."""
       self._loco.Move(0, speed, 0)


   def move_rotate(self, speed: float = 0.3) -> None:
       """Rotaciona no lugar (positivo = anti-horário). speed em rad/s."""
       self._loco.Move(0, 0, speed)


   def stop(self) -> None:
       """Para o movimento enviando velocidade zero."""
       self._loco.Move(0, 0, 0)


   # ------------------------------------------------------------------
   # Locomoção — gestos com o corpo
   # ------------------------------------------------------------------


   def wave_hand(self, turn_around: bool = False) -> None:
       """
       Acena com a mão.


       Args:
           turn_around: Se True, acena girando o corpo (wave hand 2).
       """
       self._loco.WaveHand(turn_around)


   def loco_shake_hand(self) -> None:
       """Aperto de mão via LocoClient (executa duas vezes com pausa)."""
       self._loco.ShakeHand()
       time.sleep(3)
       self._loco.ShakeHand()




from embodied_ai import EmbodiedAI
import time


robot = EmbodiedAI("eth0")


detections = robot.detect_with_distance("person")
print("Rodando...")


while True:
   detections = robot.detect_with_distance("person")
   if any(d["distance_m"] > 0.0 and d["distance_m"] < 1.0 for d in detections):
       robot.clap()
   time.sleep(1)

