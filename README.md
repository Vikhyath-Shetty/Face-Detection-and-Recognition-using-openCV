# Face Detection and Recognition

A Python project for real-time face detection and recognition using OpenCV. It uses the Haar Cascade method for face detection and the LBPH (Local Binary Patterns Histograms) algorithm for face recognition. This tool allows you to capture face images, train a recognition model, and perform real-time face recognition from a webcam or IP camera stream.

---

## Features

- **Face Capture:** Collect face images for different people using your webcam or an IP camera.
- **Model Training:** Train a face recognition model (LBPH) on the collected dataset.
- **Real-Time Recognition:** Recognize faces in real-time from a video stream.
- **Easy CLI:** Simple command-line interface for all operations.

---

## Requirements

- Python 3.8+
- OpenCV (with `opencv-contrib-python` for face recognition)
- NumPy

Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv pip install opencv-contrib-python numpy
```

---

## Usage

Run the main script with the desired mode:

```bash
python main.py <mode> [--camera <source>]
```

### Modes

- `capture` : Collect face images for a person.
- `train` : Train the recognition model on the collected dataset.
- `detect` : Run real-time face recognition.

### Camera Source

- Default: `0` (your primary webcam)
- For IP cameras: Pass the RTSP/HTTP URL.

#### Running with `uv`

You can use [`uv`](https://github.com/astral-sh/uv) to run the scripts instead of `python`:

```bash
uv run main.py <mode> [--camera <source>]
```

### Examples

**Capture faces from webcam:**

```bash
uv run main.py capture
```

**Capture faces from an IP camera:**

```bash
uv run main.py capture --camera http://<ip>:<port>/stream
```

**Train the model:**

```bash
uv run main.py train
```

**Run real-time recognition:**

```bash
uv run main.py detect
```

**Capture faces from webcam:**

```bash
uv run main.py capture
```

**Capture faces from an IP camera:**

```bash
uv run main.py capture --camera http://<ip>:<port>/stream
```

**Train the model:**

```bash
python main.py train
```

**Run real-time recognition:**

```bash
python main.py detect
```

---

## Retraining

To retrain from scratch, remove the `dataset` and `model` folders, then repeat face capture and training:

```bash
rm -rf dataset model
uv run main.py capture
uv run main.py train
```

---

## Project Structure

```
Face_Detection_and_Recognition/
│
├── main.py
├── src/
│   ├── capture_faces.py
│   ├── train_faces.py
│   ├── recognize_faces.py
│   ├── utils.py
│
├── dataset/      # Collected face images (auto-created)
├── model/        # Trained model and label map (auto-created)
└── README.md
```

---

## Notes

- The first time you run `capture`, you will be prompted to enter the person's name. 20 images will be collected per person.
- Every time you capture a new person, you must retrain the model by running `train` before using recognition.
- After capturing faces for all people, run `train` to build or update the recognition model.
- Use `detect` to recognize faces in real-time.
- Press `q` to quit the video window at any time.

---

## Troubleshooting

- **cv2.data.haarcascades error:**  
  If you get an error about `cv2.data.haarcascades`, ensure you have `opencv-contrib-python` installed. Alternatively, download the Haar cascade XML file and place it in your project directory.

- **No module named 'cv2.face':**  
  Make sure you installed `opencv-contrib-python`, not just `opencv-python`.

---

## License

This project is for educational purposes.

---

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [LBPH Face Recognizer](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_face_detection.html)
