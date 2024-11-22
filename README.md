# People_counter
Here’s a tailored **README.md** file for your people detection project. It’s structured similarly to your car counting README but adapted for detecting and counting people instead.

---

# 🧍 People Detection and Counting using YOLOv8

This project implements a **People Detection and Counting System** using the YOLOv8 object detection model. It detects and tracks people in a video feed, counting the number of individuals as they pass through a designated area.

---

## ✨ Features

- **Real-time object detection** to detect people in video frames.
- **Accurate tracking** of individuals across frames to prevent double-counting.
- **Customizable detection area** for monitoring specific regions.
- Displays the **total count of detected people** on the output video.

---

## 📂 Project Structure

| File        | Description                                                              |
|-------------|--------------------------------------------------------------------------|
| `main.py`   | Core script for detection, counting, and displaying the output.          |
| `sort.py`   | Implements the tracking algorithm to uniquely identify people in frames. |
| `graphics.png` | Optional graphical assets for UI enhancements (e.g., overlays).        |
| `mask.png`  | Mask image to define the region of interest for people counting.         |

---

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone <https://github.com/uttuJod/People_counter>
   cd PeopleCounter
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python numpy
   ```

3. Run the project:
   ```bash
   python main.py
   ```

---

## 🎥 How It Works

1. **Object Detection:**  
   The YOLOv8 model detects people in each frame of the input video.

2. **Object Tracking:**  
   The `sort.py` file assigns unique IDs to each detected person and tracks them across frames to ensure accurate counting.

3. **Counting People:**  
   A line or region is defined in the frame, and the system increments the count whenever a person crosses this area.

---

## 🖼️ Example Output

![People Detection Example](graphics.png)  
*()*

---

## 🙏 Acknowledgments

- **SORT Algorithm:** The `sort.py` implementation is adapted from an external source to enable robust tracking.

---

Let me know if you'd like to add a demo video, a GIF, or more details!
