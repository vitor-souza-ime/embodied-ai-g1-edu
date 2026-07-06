# =============================================================================
# EXEMPLO 1 — Básico
# Detectar pessoa e acenar (high wave)
# =============================================================================

# ── COM framework ─────────────────────────────────────────────────────────────
from embodied_ai import EmbodiedAI
import time

robot = EmbodiedAI("eth0")

print("Aguardando pessoa...")
while True:
    detections = robot.detect()
    if any(d["class"] == "person" for d in detections):
        print("Pessoa detectada! Acenando.")
        robot.high_wave()
    time.sleep(1)


# ── SEM framework ─────────────────────────────────────────────────────────────
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map

ChannelFactoryInitialize(0, "eth0")
arm = G1ArmActionClient()
arm.SetTimeout(10.0)
arm.Init()

yolo = YOLO("yolo26m.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("Aguardando pessoa...")
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        time.sleep(1)
        continue
    img = np.asanyarray(color_frame.get_data())
    results = yolo(img, verbose=False)
    for result in results:
        for box in result.boxes:
            if float(box.conf[0]) >= 0.5:
                name = yolo.names[int(box.cls[0])]
                if name == "person":
                    print("Pessoa detectada! Acenando.")
                    arm.ExecuteAction(action_map.get("high wave"))
    time.sleep(1)


# =============================================================================
# EXEMPLO 2 — Básico
# Medir distância de pessoa e exibir no terminal
# =============================================================================

# ── COM framework ─────────────────────────────────────────────────────────────
from embodied_ai import EmbodiedAI

robot = EmbodiedAI("eth0")

for _ in range(10):
    dist = robot.object_distance("person")
    if dist is not None:
        print(f"Distância até a pessoa: {dist:.2f} m")
    else:
        print("Nenhuma pessoa detectada.")
    import time; time.sleep(1)


# ── SEM framework ─────────────────────────────────────────────────────────────
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

ChannelFactoryInitialize(0, "eth0")

yolo = YOLO("yolo26m.pt")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

JANELA = 10

for _ in range(10):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        time.sleep(1)
        continue

    img = np.asanyarray(color_frame.get_data())
    depth_arr = np.asanyarray(depth_frame.get_data()).astype(float)
    results = yolo(img, verbose=False)

    distances = []
    for result in results:
        for box in result.boxes:
            if float(box.conf[0]) < 0.5:
                continue
            if yolo.names[int(box.cls[0])] != "person":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            region = depth_arr[
                max(0, cy - JANELA):min(480, cy + JANELA),
                max(0, cx - JANELA):min(640, cx + JANELA),
            ].copy()
            region[region == 0] = np.nan
            if not np.all(np.isnan(region)):
                distances.append(float(np.nanmedian(region)) * depth_frame.get_units())

    if distances:
        print(f"Distância até a pessoa: {min(distances):.2f} m")
    else:
        print("Nenhuma pessoa detectada.")
    time.sleep(1)


# =============================================================================
# EXEMPLO 3 — Básico
# Levantar da posição deitado e assumir postura em pé
# =============================================================================

# ── COM framework ─────────────────────────────────────────────────────────────
from embodied_ai import EmbodiedAI
import time

robot = EmbodiedAI("eth0")

print("Levantando o robô...")
robot.lie_to_standup()
time.sleep(3)
robot.high_stand()
print("Robô em pé.")


# ── SEM framework ─────────────────────────────────────────────────────────────
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

ChannelFactoryInitialize(0, "eth0")
loco = LocoClient()
loco.SetTimeout(10.0)
loco.Init()

print("Levantando o robô...")
loco.Damp()
time.sleep(0.5)
loco.Lie2StandUp()
time.sleep(3)
loco.HighStand()
print("Robô em pé.")


# =============================================================================
# EXEMPLO 4 — Intermediário
# Embodied AI: aproximar-se de pessoa e cumprimentar com aperto de mão
# Percepção → distância → locomoção → gesto
# =============================================================================

# ── COM framework ─────────────────────────────────────────────────────────────
from embodied_ai import EmbodiedAI
import time

robot = EmbodiedAI("eth0")

print("Procurando pessoa para cumprimentar...")
while True:
    dist = robot.object_distance("person")

    if dist is None:
        # Nenhuma pessoa: girar devagar para procurar
        robot.move_rotate(0.2)
        time.sleep(0.5)
        robot.stop()
    elif dist > 1.2:
        # Pessoa longe: avançar
        robot.move_forward(0.3)
        time.sleep(0.4)
        robot.stop()
    else:
        # Pessoa próxima: cumprimentar e encerrar
        robot.stop()
        print(f"Pessoa a {dist:.2f} m — cumprimentando!")
        robot.shake_hand()
        break

    time.sleep(0.3)


# ── SEM framework ─────────────────────────────────────────────────────────────
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

ChannelFactoryInitialize(0, "eth0")
arm = G1ArmActionClient(); arm.SetTimeout(10.0); arm.Init()
loco = LocoClient(); loco.SetTimeout(10.0); loco.Init()
yolo = YOLO("yolo26m.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
JANELA = 10

def medir_distancia(classe):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    if not cf or not df:
        return None
    img = np.asanyarray(cf.get_data())
    depth = np.asanyarray(df.get_data()).astype(float)
    for res in yolo(img, verbose=False):
        for box in res.boxes:
            if float(box.conf[0]) < 0.5: continue
            if yolo.names[int(box.cls[0])] != classe: continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx,cy = (x1+x2)//2,(y1+y2)//2
            reg = depth[max(0,cy-JANELA):min(480,cy+JANELA),
                        max(0,cx-JANELA):min(640,cx+JANELA)].copy()
            reg[reg==0] = np.nan
            if not np.all(np.isnan(reg)):
                return float(np.nanmedian(reg)) * df.get_units()
    return None

print("Procurando pessoa para cumprimentar...")
while True:
    dist = medir_distancia("person")
    if dist is None:
        loco.Move(0, 0, 0.2); time.sleep(0.5); loco.Move(0,0,0)
    elif dist > 1.2:
        loco.Move(0.3, 0, 0); time.sleep(0.4); loco.Move(0,0,0)
    else:
        loco.Move(0,0,0)
        print(f"Pessoa a {dist:.2f} m — cumprimentando!")
        arm.ExecuteAction(action_map.get("shake hand"))
        time.sleep(2)
        arm.ExecuteAction(action_map.get("release arm"))
        break
    time.sleep(0.3)


# =============================================================================
# EXEMPLO 5 — Intermediário
# Embodied AI: reação emocional baseada em distância
# Percepção → classificação de proximidade → gesto expressivo
# =============================================================================

# ── COM framework ─────────────────────────────────────────────────────────────
from embodied_ai import EmbodiedAI
import time

robot = EmbodiedAI("eth0")

print("Modo de interação social ativo.")
try:
    while True:
        detections = robot.detect_with_distance("person")
        if not detections:
            time.sleep(0.5)
            continue

        # Pega a pessoa mais próxima
        closest = min(detections, key=lambda d: d["distance_m"] if d["distance_m"] > 0 else 9999)
        dist = closest["distance_m"]

        if 0 < dist < 0.8:
            print(f"[{dist:.2f}m] Muito próximo! → Abraço")
            robot.hug()
            time.sleep(3)
        elif 0.8 <= dist < 1.5:
            print(f"[{dist:.2f}m] Zona social → Aperto de mão")
            robot.shake_hand()
            time.sleep(2)
        elif 1.5 <= dist < 3.0:
            print(f"[{dist:.2f}m] Zona pública → Aceno")
            robot.high_wave()
            time.sleep(1)
        else:
            print(f"[{dist:.2f}m] Longe — aguardando.")

        time.sleep(1.5)
except KeyboardInterrupt:
    robot.release_arm()
    robot.stop()
    robot.close_camera()


# ── SEM framework ─────────────────────────────────────────────────────────────
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map

ChannelFactoryInitialize(0, "eth0")
arm = G1ArmActionClient(); arm.SetTimeout(10.0); arm.Init()
yolo = YOLO("yolo26m.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
JANELA = 10

def detectar_pessoas_com_distancia():
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    if not cf or not df: return []
    img = np.asanyarray(cf.get_data())
    depth = np.asanyarray(df.get_data()).astype(float)
    result_list = []
    for res in yolo(img, verbose=False):
        for box in res.boxes:
            if float(box.conf[0]) < 0.5: continue
            if yolo.names[int(box.cls[0])] != "person": continue
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cx,cy = (x1+x2)//2,(y1+y2)//2
            reg = depth[max(0,cy-JANELA):min(480,cy+JANELA),
                        max(0,cx-JANELA):min(640,cx+JANELA)].copy()
            reg[reg==0]=np.nan
            dist = float(np.nanmedian(reg))*df.get_units() if not np.all(np.isnan(reg)) else 0.0
            result_list.append({"distance_m": dist})
    return result_list

print("Modo de interação social ativo.")
try:
    while True:
        detections = detectar_pessoas_com_distancia()
        if not detections:
            time.sleep(0.5); continue
        closest = min(detections, key=lambda d: d["distance_m"] if d["distance_m"]>0 else 9999)
        dist = closest["distance_m"]
        if 0 < dist < 0.8:
            print(f"[{dist:.2f}m] Muito próximo! → Abraço")
            arm.ExecuteAction(action_map.get("hug")); time.sleep(3)
            arm.ExecuteAction(action_map.get("release arm"))
        elif 0.8 <= dist < 1.5:
            print(f"[{dist:.2f}m] Zona social → Aperto de mão")
            arm.ExecuteAction(action_map.get("shake hand")); time.sleep(2)
            arm.ExecuteAction(action_map.get("release arm"))
        elif 1.5 <= dist < 3.0:
            print(f"[{dist:.2f}m] Zona pública → Aceno")
            arm.ExecuteAction(action_map.get("high wave")); time.sleep(1)
        else:
            print(f"[{dist:.2f}m] Longe — aguardando.")
        time.sleep(1.5)
except KeyboardInterrupt:
    arm.ExecuteAction(action_map.get("release arm"))
    pipeline.stop()


# =============================================================================
# EXEMPLO 6 — Intermediário
# Embodied AI: patrulha autônoma — avança, detecta obstáculo, desvia
# Percepção de distância → controle reativo de locomoção
# =============================================================================

# ── COM framework ─────────────────────────────────────────────────────────────
from embodied_ai import EmbodiedAI
import time

robot = EmbodiedAI("eth0")
DIST_PARADA = 0.8   # metros

print("Patrulha iniciada.")
try:
    while True:
        dist = robot.object_distance()   # qualquer objeto
        if dist is not None and dist < DIST_PARADA:
            print(f"Obstáculo a {dist:.2f}m! Desviando...")
            robot.stop()
            robot.move_rotate(0.5)     # gira ~90° (depende do tempo)
            time.sleep(1.6)
            robot.stop()
        else:
            robot.move_forward(0.2)
        time.sleep(0.3)
except KeyboardInterrupt:
    robot.stop()
    robot.close_camera()


# ── SEM framework ─────────────────────────────────────────────────────────────
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

ChannelFactoryInitialize(0, "eth0")
loco = LocoClient(); loco.SetTimeout(10.0); loco.Init()
yolo = YOLO("yolo26m.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
JANELA = 10
DIST_PARADA = 0.8

def menor_distancia():
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    if not cf or not df: return None
    img = np.asanyarray(cf.get_data())
    depth = np.asanyarray(df.get_data()).astype(float)
    distancias = []
    for res in yolo(img, verbose=False):
        for box in res.boxes:
            if float(box.conf[0]) < 0.5: continue
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            cx,cy=(x1+x2)//2,(y1+y2)//2
            reg = depth[max(0,cy-JANELA):min(480,cy+JANELA),
                        max(0,cx-JANELA):min(640,cx+JANELA)].copy()
            reg[reg==0]=np.nan
            if not np.all(np.isnan(reg)):
                distancias.append(float(np.nanmedian(reg))*df.get_units())
    return round(min(distancias),3) if distancias else None

print("Patrulha iniciada.")
try:
    while True:
        dist = menor_distancia()
        if dist is not None and dist < DIST_PARADA:
            print(f"Obstáculo a {dist:.2f}m! Desviando...")
            loco.Move(0,0,0); time.sleep(0.1)
            loco.Move(0,0,0.5); time.sleep(1.6)
            loco.Move(0,0,0)
        else:
            loco.Move(0.2,0,0)
        time.sleep(0.3)
except KeyboardInterrupt:
    loco.Move(0,0,0)
    pipeline.stop()


# =============================================================================
# EXEMPLO 7 — Intermediário
# Embodied AI: contar pessoas no ambiente e reagir ao número delas
# Percepção múltipla → contagem → seleção de gesto por quantidade
# =============================================================================

# ── COM framework ─────────────────────────────────────────────────────────────
from embodied_ai import EmbodiedAI
import time

robot = EmbodiedAI("eth0")

print("Analisando plateia...")
while True:
    detections = robot.detect_with_distance("person")
    validas = [d for d in detections if d["distance_m"] > 0]
    n = len(validas)
    print(f"{n} pessoa(s) detectada(s).")

    if n == 0:
        pass  # nada
    elif n == 1:
        robot.face_wave()
    elif n == 2:
        robot.high_five()
    else:
        # Multidão: palmas + beijo com ambas as mãos
        robot.clap()
        time.sleep(1)
        robot.two_hand_kiss()

    time.sleep(3)


# ── SEM framework ─────────────────────────────────────────────────────────────
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map

ChannelFactoryInitialize(0, "eth0")
arm = G1ArmActionClient(); arm.SetTimeout(10.0); arm.Init()
yolo = YOLO("yolo26m.pt")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
JANELA=10

def detectar_n_pessoas():
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    cf = aligned.get_color_frame(); df = aligned.get_depth_frame()
    if not cf or not df: return []
    img = np.asanyarray(cf.get_data())
    depth = np.asanyarray(df.get_data()).astype(float)
    pessoas = []
    for res in yolo(img, verbose=False):
        for box in res.boxes:
            if float(box.conf[0]) < 0.5: continue
            if yolo.names[int(box.cls[0])] != "person": continue
            x1,y1,x2,y2=map(int,box.xyxy[0]); cx,cy=(x1+x2)//2,(y1+y2)//2
            reg=depth[max(0,cy-JANELA):min(480,cy+JANELA),
                      max(0,cx-JANELA):min(640,cx+JANELA)].copy()
            reg[reg==0]=np.nan
            d=float(np.nanmedian(reg))*df.get_units() if not np.all(np.isnan(reg)) else 0.0
            if d>0: pessoas.append(d)
    return pessoas

print("Analisando plateia...")
while True:
    pessoas = detectar_n_pessoas()
    n = len(pessoas)
    print(f"{n} pessoa(s) detectada(s).")
    if n == 1:
        arm.ExecuteAction(action_map.get("face wave"))
    elif n == 2:
        arm.ExecuteAction(action_map.get("high five")); time.sleep(2)
        arm.ExecuteAction(action_map.get("release arm"))
    elif n >= 3:
        arm.ExecuteAction(action_map.get("clap")); time.sleep(1)
        arm.ExecuteAction(action_map.get("two-hand kiss"))
    time.sleep(3)


# =============================================================================
# EXEMPLO 8 — Avançado
# Embodied AI: seguir pessoa mantendo distância-alvo (controle proporcional)
# Percepção contínua → erro de distância → controle P de velocidade
# =============================================================================

# ── COM framework ─────────────────────────────────────────────────────────────
from embodied_ai import EmbodiedAI
import time

robot = EmbodiedAI("eth0")
DIST_ALVO = 1.2    # metros
KP        = 0.4    # ganho proporcional
V_MAX     = 0.5    # velocidade máxima (m/s)

print("Seguindo pessoa (Ctrl+C para parar)...")
try:
    while True:
        dist = robot.object_distance("person")
        if dist is None:
            robot.stop()
            time.sleep(0.3)
            continue

        erro = dist - DIST_ALVO
        vx = max(-V_MAX, min(V_MAX, KP * erro))  # clamp

        if abs(erro) < 0.05:
            robot.stop()
            print(f"Na distância-alvo ({dist:.2f}m).")
        else:
            robot.move(vx, 0, 0)
            print(f"dist={dist:.2f}m  erro={erro:+.2f}m  vx={vx:+.2f}m/s")

        time.sleep(0.2)
except KeyboardInterrupt:
    robot.stop()
    robot.close_camera()


# ── SEM framework ─────────────────────────────────────────────────────────────
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

ChannelFactoryInitialize(0, "eth0")
loco = LocoClient(); loco.SetTimeout(10.0); loco.Init()
yolo = YOLO("yolo26m.pt")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)
JANELA=10; DIST_ALVO=1.2; KP=0.4; V_MAX=0.5

def dist_pessoa():
    frames=pipeline.wait_for_frames(); aligned=align.process(frames)
    cf=aligned.get_color_frame(); df=aligned.get_depth_frame()
    if not cf or not df: return None
    img=np.asanyarray(cf.get_data())
    depth=np.asanyarray(df.get_data()).astype(float)
    dists=[]
    for res in yolo(img,verbose=False):
        for box in res.boxes:
            if float(box.conf[0])<0.5: continue
            if yolo.names[int(box.cls[0])]!="person": continue
            x1,y1,x2,y2=map(int,box.xyxy[0]); cx,cy=(x1+x2)//2,(y1+y2)//2
            reg=depth[max(0,cy-JANELA):min(480,cy+JANELA),
                      max(0,cx-JANELA):min(640,cx+JANELA)].copy()
            reg[reg==0]=np.nan
            if not np.all(np.isnan(reg)):
                dists.append(float(np.nanmedian(reg))*df.get_units())
    return round(min(dists),3) if dists else None

print("Seguindo pessoa (Ctrl+C para parar)...")
try:
    while True:
        dist=dist_pessoa()
        if dist is None:
            loco.Move(0,0,0); time.sleep(0.3); continue
        erro=dist-DIST_ALVO
        vx=max(-V_MAX,min(V_MAX,KP*erro))
        if abs(erro)<0.05:
            loco.Move(0,0,0); print(f"Na distância-alvo ({dist:.2f}m).")
        else:
            loco.Move(vx,0,0)
            print(f"dist={dist:.2f}m  erro={erro:+.2f}m  vx={vx:+.2f}m/s")
        time.sleep(0.2)
except KeyboardInterrupt:
    loco.Move(0,0,0); pipeline.stop()


# =============================================================================
# EXEMPLO 9 — Avançado
# Embodied AI: alinhamento lateral com pessoa usando controle angular (yaw)
# Percepção de posição horizontal (cx) → erro de centramento → controle P de yaw
# =============================================================================

# ── COM framework ─────────────────────────────────────────────────────────────
from embodied_ai import EmbodiedAI
import time

robot = EmbodiedAI("eth0")
IMG_CX   = 320       # centro horizontal da imagem (640px)
KP_YAW   = 0.003     # ganho proporcional para yaw
YAW_MAX  = 0.4       # rad/s máximo

print("Alinhando com a pessoa (Ctrl+C para parar)...")
try:
    while True:
        detections = robot.detect_with_distance("person")
        if not detections:
            robot.move_rotate(0.2)   # gira lentamente à procura
            time.sleep(0.3)
            continue

        # Pessoa mais próxima com depth válida
        validas = [d for d in detections if d["distance_m"] > 0]
        if not validas:
            time.sleep(0.3)
            continue
        alvo = min(validas, key=lambda d: d["distance_m"])

        cx_obj = alvo["center"][0]
        erro_x = cx_obj - IMG_CX           # positivo → objeto à direita
        vyaw   = -KP_YAW * erro_x          # gira para apontar ao centro
        vyaw   = max(-YAW_MAX, min(YAW_MAX, vyaw))

        if abs(erro_x) < 20:
            robot.stop()
            print(f"Centrado! cx={cx_obj}  dist={alvo['distance_m']:.2f}m")
        else:
            robot.move(0, 0, vyaw)
            print(f"cx={cx_obj}  erro_x={erro_x:+d}  vyaw={vyaw:+.3f}")

        time.sleep(0.15)
except KeyboardInterrupt:
    robot.stop()
    robot.close_camera()


# ── SEM framework ─────────────────────────────────────────────────────────────
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

ChannelFactoryInitialize(0, "eth0")
loco=LocoClient(); loco.SetTimeout(10.0); loco.Init()
yolo=YOLO("yolo26m.pt")
pipeline=rs.pipeline()
config=rs.config()
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
pipeline.start(config)
align=rs.align(rs.stream.color)
JANELA=10; IMG_CX=320; KP_YAW=0.003; YAW_MAX=0.4

def detectar_com_cx():
    frames=pipeline.wait_for_frames(); aligned=align.process(frames)
    cf=aligned.get_color_frame(); df=aligned.get_depth_frame()
    if not cf or not df: return []
    img=np.asanyarray(cf.get_data())
    depth=np.asanyarray(df.get_data()).astype(float)
    lista=[]
    for res in yolo(img,verbose=False):
        for box in res.boxes:
            if float(box.conf[0])<0.5: continue
            if yolo.names[int(box.cls[0])]!="person": continue
            x1,y1,x2,y2=map(int,box.xyxy[0]); cx,cy=(x1+x2)//2,(y1+y2)//2
            reg=depth[max(0,cy-JANELA):min(480,cy+JANELA),
                      max(0,cx-JANELA):min(640,cx+JANELA)].copy()
            reg[reg==0]=np.nan
            d=float(np.nanmedian(reg))*df.get_units() if not np.all(np.isnan(reg)) else 0.0
            lista.append({"cx":cx,"distance_m":d})
    return lista

print("Alinhando com a pessoa (Ctrl+C para parar)...")
try:
    while True:
        detections=detectar_com_cx()
        validas=[d for d in detections if d["distance_m"]>0]
        if not validas:
            loco.Move(0,0,0.2); time.sleep(0.3); continue
        alvo=min(validas,key=lambda d:d["distance_m"])
        erro_x=alvo["cx"]-IMG_CX
        vyaw=max(-YAW_MAX,min(YAW_MAX,-KP_YAW*erro_x))
        if abs(erro_x)<20:
            loco.Move(0,0,0)
            print(f"Centrado! cx={alvo['cx']}  dist={alvo['distance_m']:.2f}m")
        else:
            loco.Move(0,0,vyaw)
            print(f"cx={alvo['cx']}  erro_x={erro_x:+d}  vyaw={vyaw:+.3f}")
        time.sleep(0.15)
except KeyboardInterrupt:
    loco.Move(0,0,0); pipeline.stop()


# =============================================================================
# EXEMPLO 10 — Avançado
# Embodied AI: máquina de estados social completa
# BUSCA → APROXIMAÇÃO → ALINHAMENTO → SAUDAÇÃO → RECUO → IDLE
# Percepção contínua + controle P linear/angular + gesto expressivo
# =============================================================================

# ── COM framework ─────────────────────────────────────────────────────────────
from embodied_ai import EmbodiedAI
import time
from enum import Enum, auto

class Estado(Enum):
    BUSCA       = auto()
    APROXIMACAO = auto()
    ALINHAMENTO = auto()
    SAUDACAO    = auto()
    RECUO       = auto()
    IDLE        = auto()

robot = EmbodiedAI("eth0")

IMG_CX      = 320
DIST_SAUDA  = 1.0    # metros para iniciar saudação
DIST_SEGURA = 2.0    # metros para recuar antes de buscar novamente
KP_VX       = 0.35
KP_YAW      = 0.003
YAW_MAX     = 0.4
VX_MAX      = 0.4
IDLE_SECS   = 5

estado = Estado.BUSCA
t_idle = None

print("Máquina de estados social iniciada. Ctrl+C para sair.")
try:
    while True:
        detections = robot.detect_with_distance("person")
        validas    = [d for d in detections if d["distance_m"] > 0]

        if estado == Estado.BUSCA:
            if not validas:
                robot.move_rotate(0.25)
            else:
                print("→ APROXIMAÇÃO")
                estado = Estado.APROXIMACAO

        elif estado == Estado.APROXIMACAO:
            if not validas:
                estado = Estado.BUSCA
            else:
                alvo  = min(validas, key=lambda d: d["distance_m"])
                dist  = alvo["distance_m"]
                cx    = alvo["center"][0]
                erro_x = cx - IMG_CX
                vyaw  = max(-YAW_MAX, min(YAW_MAX, -KP_YAW * erro_x))
                erro_d = dist - DIST_SAUDA
                vx    = max(-VX_MAX, min(VX_MAX, KP_VX * erro_d))
                robot.move(vx, 0, vyaw)
                print(f"  APROXIMAÇÃO  dist={dist:.2f}m  cx={cx}  vx={vx:+.2f}  vyaw={vyaw:+.3f}")
                if abs(erro_x) < 30 and dist < DIST_SAUDA + 0.1:
                    robot.stop()
                    print("→ ALINHAMENTO FINO")
                    estado = Estado.ALINHAMENTO

        elif estado == Estado.ALINHAMENTO:
            if not validas:
                estado = Estado.BUSCA
            else:
                alvo   = min(validas, key=lambda d: d["distance_m"])
                cx     = alvo["center"][0]
                erro_x = cx - IMG_CX
                vyaw   = max(-YAW_MAX, min(YAW_MAX, -KP_YAW * erro_x))
                if abs(erro_x) < 15:
                    robot.stop()
                    print("→ SAUDAÇÃO")
                    estado = Estado.SAUDACAO
                else:
                    robot.move(0, 0, vyaw)

        elif estado == Estado.SAUDACAO:
            robot.shake_hand()
            time.sleep(1)
            robot.clap()
            time.sleep(1)
            robot.heart()
            print("→ RECUO")
            estado = Estado.RECUO

        elif estado == Estado.RECUO:
            # Afasta-se até distância segura
            dist = robot.object_distance("person") or 0.0
            if dist < DIST_SEGURA:
                robot.move(-0.25, 0, 0)
            else:
                robot.stop()
                t_idle = time.time()
                print("→ IDLE")
                estado = Estado.IDLE

        elif estado == Estado.IDLE:
            robot.stop()
            if time.time() - t_idle > IDLE_SECS:
                print("→ BUSCA")
                estado = Estado.BUSCA

        time.sleep(0.2)

except KeyboardInterrupt:
    robot.stop()
    robot.release_arm()
    robot.close_camera()
    print("Encerrado.")


# ── SEM framework ─────────────────────────────────────────────────────────────
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from enum import Enum, auto
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

ChannelFactoryInitialize(0, "eth0")
arm=G1ArmActionClient(); arm.SetTimeout(10.0); arm.Init()
loco=LocoClient(); loco.SetTimeout(10.0); loco.Init()
yolo=YOLO("yolo26m.pt")
pipeline=rs.pipeline()
config=rs.config()
config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
config.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
pipeline.start(config)
align=rs.align(rs.stream.color)

JANELA=10; IMG_CX=320; DIST_SAUDA=1.0; DIST_SEGURA=2.0
KP_VX=0.35; KP_YAW=0.003; YAW_MAX=0.4; VX_MAX=0.4; IDLE_SECS=5

class Estado(Enum):
    BUSCA=auto(); APROXIMACAO=auto(); ALINHAMENTO=auto()
    SAUDACAO=auto(); RECUO=auto(); IDLE=auto()

def capturar():
    frames=pipeline.wait_for_frames(); aligned=align.process(frames)
    cf=aligned.get_color_frame(); df=aligned.get_depth_frame()
    if not cf or not df: return None,None
    return np.asanyarray(cf.get_data()),df

def mediana_depth(df,cx,cy):
    j=JANELA; depth=np.asanyarray(df.get_data()).astype(float)
    reg=depth[max(0,cy-j):min(480,cy+j),max(0,cx-j):min(640,cx+j)].copy()
    reg[reg==0]=np.nan
    if np.all(np.isnan(reg)): return 0.0
    return float(np.nanmedian(reg))*df.get_units()

def detectar_pessoas():
    img,df=capturar()
    if img is None: return []
    lista=[]
    for res in yolo(img,verbose=False):
        for box in res.boxes:
            if float(box.conf[0])<0.5: continue
            if yolo.names[int(box.cls[0])]!="person": continue
            x1,y1,x2,y2=map(int,box.xyxy[0]); cx,cy=(x1+x2)//2,(y1+y2)//2
            d=mediana_depth(df,cx,cy)
            lista.append({"cx":cx,"distance_m":d})
    return [p for p in lista if p["distance_m"]>0]

estado=Estado.BUSCA; t_idle=None

print("Máquina de estados social iniciada. Ctrl+C para sair.")
try:
    while True:
        validas=detectar_pessoas()

        if estado==Estado.BUSCA:
            if not validas: loco.Move(0,0,0.25)
            else: print("→ APROXIMAÇÃO"); estado=Estado.APROXIMACAO

        elif estado==Estado.APROXIMACAO:
            if not validas: estado=Estado.BUSCA
            else:
                alvo=min(validas,key=lambda d:d["distance_m"])
                dist=alvo["distance_m"]; cx=alvo["cx"]
                erro_x=cx-IMG_CX; vyaw=max(-YAW_MAX,min(YAW_MAX,-KP_YAW*erro_x))
                vx=max(-VX_MAX,min(VX_MAX,KP_VX*(dist-DIST_SAUDA)))
                loco.Move(vx,0,vyaw)
                print(f"  APROXIMAÇÃO  dist={dist:.2f}m  cx={cx}  vx={vx:+.2f}  vyaw={vyaw:+.3f}")
                if abs(erro_x)<30 and dist<DIST_SAUDA+0.1:
                    loco.Move(0,0,0); print("→ ALINHAMENTO"); estado=Estado.ALINHAMENTO

        elif estado==Estado.ALINHAMENTO:
            if not validas: estado=Estado.BUSCA
            else:
                alvo=min(validas,key=lambda d:d["distance_m"]); cx=alvo["cx"]
                erro_x=cx-IMG_CX; vyaw=max(-YAW_MAX,min(YAW_MAX,-KP_YAW*erro_x))
                if abs(erro_x)<15:
                    loco.Move(0,0,0); print("→ SAUDAÇÃO"); estado=Estado.SAUDACAO
                else: loco.Move(0,0,vyaw)

        elif estado==Estado.SAUDACAO:
            arm.ExecuteAction(action_map.get("shake hand")); time.sleep(2)
            arm.ExecuteAction(action_map.get("release arm")); time.sleep(0.5)
            arm.ExecuteAction(action_map.get("clap")); time.sleep(1)
            arm.ExecuteAction(action_map.get("heart")); time.sleep(2)
            arm.ExecuteAction(action_map.get("release arm"))
            print("→ RECUO"); estado=Estado.RECUO

        elif estado==Estado.RECUO:
            validas2=detectar_pessoas()
            dist=min((p["distance_m"] for p in validas2),default=DIST_SEGURA+1)
            if dist<DIST_SEGURA: loco.Move(-0.25,0,0)
            else:
                loco.Move(0,0,0); t_idle=time.time(); print("→ IDLE"); estado=Estado.IDLE

        elif estado==Estado.IDLE:
            loco.Move(0,0,0)
            if time.time()-t_idle>IDLE_SECS: print("→ BUSCA"); estado=Estado.BUSCA

        time.sleep(0.2)

except KeyboardInterrupt:
    loco.Move(0,0,0)
    arm.ExecuteAction(action_map.get("release arm"))
    pipeline.stop()
    print("Encerrado.")
