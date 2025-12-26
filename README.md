# **Altersklassifizierung mit Deep Learning**

Eine Anwendung zur automatischen Alterskllassifizierung von Gesichtern mit PyTorch und einer RTX 2050 GPU. Trainiert auf dem UTKFace-Datensatz mit 65.1% Genauigkeit.

![Altersklassifizierung](https://img.shields.io/badge/DeepLearning-PyTorch-red)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![GPU](https://img.shields.io/badge/GPU-RTX2050-green)
![Accuracy](https://img.shields.io/badge/Accuracy-65.1%25-brightgreen)

## ** Übersicht**

Diese Anwendung kann:
-  Gesichter in Bildern erkennen
-  Das Alter in 5 Klassen einteilen (0-17, 18-29, 30-44, 45-59, 60+)
-  Live-Kamera mit Echtzeit-Analyse
-  Wahrscheinlichkeitsverteilungen anzeigen
-  Screenshots speichern
-  Batch-Verarbeitung von Bildern

## ** Schnellstart**

### **1. Voraussetzungen**
- Windows 10/11
- Python 3.10
- NVIDIA GPU mit CUDA-Unterstützung (optional)

### **2. Installation**
```bash
# 1. Repository klonen oder Ordner erstellen
cd "C:\Users\DeinName\Desktop\Altersklassifizierung"

# 2. Virtuelle Environment erstellen
py -3.10 -m venv venv

# 3. Environment aktivieren
venv\Scripts\activate

# 4. Abhängigkeiten installieren
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas numpy matplotlib opencv-python scikit-learn jupyter tqdm pillow seaborn
```

### **3. UTKFace Datensatz**
Lade den UTKFace Datensatz von Kaggle herunter und platziere alle Bilder im Ordner `UTKFace/`.

## ** Verwendung**

### **Option 1: Training starten**
```bash
python train_final_fixed.py
```
Das Training dauert ca. 60-90 Minuten und erreicht ~65% Genauigkeit.

### **Option 2: Einzelnes Bild analysieren**
```bash
python analyze_image.py
```
Wähle Option 1 und gib den Pfad zum Bild ein:
```
C:\Users\DeinName\Desktop\mein_bild.jpg
```

### **Option 3: Live-Kamera starten**
```bash
python live_camera_fixed.py
```
**Tastaturbefehle:**
- **SPACE** (Leertaste): Screenshot speichern
- **ESC**: Programm beenden
- **S**: Statistiken ein/aus
- **F**: Gesichtserkennung ein/aus

### **Option 4: Ordner analysieren**
```bash
python analyze_image.py
```
Wähle Option 3 und gib den Pfad zum Ordner an.

## ** Projektstruktur**

```
Altersklassifizierung/
├── UTKFace/                    # Trainingsdaten
├── venv/                       # Virtuelle Environment
├── train_final_fixed.py        # Haupt-Trainingsskript
├── analyze_image.py           # Bildanalyse-Tool
├── live_camera_fixed.py       # Live-Kamera
├── age_predictor.py           # Kommandozeilen-Tool
├── best_age_model.pth         # Trainiertes Modell
├── setup.bat                  # Einrichtungsskript
└── start_camera.bat           # Kamera-Startskript
```

## ** Modell-Architektur**

Das Modell verwendet eine CNN-Architektur mit:
- 4 Convolutional-Blöcke mit Batch Normalization
- 256 Neuronen in der Fully-Connected Schicht
- 5 Ausgabeklassen für Altersgruppen
- Dropout zur Vermeidung von Overfitting

**Altersgruppen:**
1. 0-17 Jahre
2. 18-29 Jahre  
3. 30-44 Jahre
4. 45-59 Jahre
5. 60+ Jahre

## ** Performance**

| Metrik | Wert |
|--------|------|
| Validation Accuracy | 65.1% |
| Trainingszeit | ~60 Minuten |
| Batch Size | 16 |
| Bildgröße | 112x112 Pixel |
| Epochen | 13 |

**Das Modell erkennt 2 von 3 Gesichtern korrekt!**

## ** Technische Details**

### **Hardware-Anforderungen**
- **Empfohlen:** NVIDIA GPU mit CUDA (RTX 2050/3050/4050)
- **Minimum:** 4GB RAM, 2GB VRAM
- **Speicher:** 5GB für UTKFace-Datensatz

### **Software-Abhängigkeiten**
```
torch==2.5.1+cu121
torchvision==0.16.1
opencv-python==4.9.0
numpy==1.26.4
pillow==10.2.0
scikit-learn==1.4.1
```

## ** Live-Kamera Features**

- **Echtzeit-Gesichtserkennung** mit OpenCV Haar Cascades
- **Farbcodierte Altersgruppen** (Grün → Rot)
- **Wahrscheinlichkeitsbalken** für alle 5 Klassen
- **FPS-Anzeige** (25-30 FPS auf RTX 2050)
- **Statistiken** über erkannte Altersgruppen
- **Automatische Screenshots** mit SPACE-Taste

## **Fehlerbehebung**

### **"Modell nicht gefunden"**
```bash
# Sicherstellen, dass Training abgeschlossen ist
python train_final_fixed.py
```

### **"No module named cv2"**
```bash
pip install opencv-python
```

### **"CUDA not available"**
- NVIDIA Treiber aktualisieren
- PyTorch mit CUDA neu installieren:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### **Kamera wird nicht erkannt**
```python
# In live_camera_fixed.py Zeile ändern:
self.cap = cv2.VideoCapture(0)  # Auf 1 oder 2 ändern
```

## ** Nächste Schritte**

### **Verbesserungsmöglichkeiten**
1. **Mehr Daten:** Erweiterter Datensatz mit mehr Altersgruppen
2. **Transfer Learning:** Vorgefertigte Modelle wie ResNet oder EfficientNet
3. **Data Augmentation:** Mehr Transformationen für bessere Generalisierung
4. **Hyperparameter-Tuning:** Learning Rate, Batch Size, Optimizer

### **Erweiterungen**
- Web-Oberfläche mit Flask/Django
- REST API für Bild-Uploads
- Mobile App mit TensorFlow Lite
- Batch-Verarbeitung für große Bildersammlungen

## ** Lizenz**

Dieses Projekt ist für Bildungszwecke konzipiert. Der UTKFace-Datensatz steht unter einer eigenen Lizenz.

## ** Danksagung**

- **UTKFace Datensatz** für die Trainingsdaten
- **PyTorch Team** für das Deep Learning Framework
- **OpenCV** für Computer Vision Funktionen


---

**Viel Spaß mit der Altersklassifizierung!**

*Letztes Training: 26. Dezember 2025*  
*Modell-Genauigkeit: 65.1%*  
*Entwickelt für RTX 2050 GPU*
