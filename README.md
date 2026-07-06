# 🤖 EmbodiedAI

### A Hardware Abstraction Layer for the Unitree G1 EDU Humanoid Robot

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Platform](https://img.shields.io/badge/Robot-Unitree_G1_EDU-orange)
![Vision](https://img.shields.io/badge/Vision-RealSense_+_YOLO-green)
![License](https://img.shields.io/badge/License-MIT-red)

---

## 🚀 Overview

EmbodiedAI is an object-oriented Python framework that provides a high-level abstraction layer for the Unitree G1 EDU humanoid robot.

The framework integrates:

📷 **Intel RealSense RGB-D Cameras**

🎯 **YOLO Object Detection**

🚶 **Locomotion Control**

🦾 **Arm Actuation**

🧠 **Embodied AI Behaviors**

Through a unified API designed for rapid development of perception-action systems.

---

## ✨ Features

- 🧩 High-level abstraction of the Unitree SDK
- 📷 Integrated RGB-D perception
- 🎯 Real-time object detection
- 🤖 Simplified robot control
- 🔄 Closed-loop perception-action workflows
- 📉 Reduced boilerplate code
- 📚 Educational and research-oriented design
- 🔧 Modular and extensible architecture

---

## 🏗️ Architecture

```text
EmbodiedAI
│
├── 👁️ Perception Layer
│   ├── RGB Stream
│   ├── Depth Stream
│   └── YOLO Detection
│
├── 🤖 Actuation Layer
│   ├── Locomotion
│   ├── Arm Control
│   └── Robot States
│
└── 🧠 High-Level API
````

---

## 📊 Experimental Results

The framework was evaluated through ten paired experiments.

| Metric                        | Improvement     |
| ----------------------------- | --------------- |
| 📉 Lines of Code (LOC)        | 51.4% reduction |
| 📉 Cyclomatic Complexity (CC) | 58.3% reduction |

These results indicate that EmbodiedAI significantly reduces implementation effort while maintaining functionality.

---

## 🛠️ Installation

```bash
git clone https://github.com/vitor-souza-ime/embodied-ai-g1-edu.git

cd embodied-ai-g1-edu

pip install -r requirements.txt
```

---

## 🎯 Example

```python
from embodied_ai import EmbodiedAI

robot = EmbodiedAI()

detections = robot.detect_objects()

for obj in detections:
    print(obj.label, obj.confidence)
```

---

## 📖 Citation

```bibtex
@article{souza2026embodiedai,
  title={EmbodiedAI: A Hardware Abstraction Layer for the Unitree G1 EDU},
  author={Souza, Vitor Amadeu and collaborators},
  year={2026}
}
```

---

## 👨‍💻 Author

**Vitor Amadeu Souza**

🏛️ Military Institute of Engineering (IME)

🇧🇷 Rio de Janeiro, Brazil

