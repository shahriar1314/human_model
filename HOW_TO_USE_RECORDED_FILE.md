# Recording and Processing ZED Body Tracking Data (SVO Workflow)

This document describes how to:

1. Record a ZED camera session to an **SVO/SVO2 file**
2. Run the **BODY_18 human detector** on the recorded file instead of a live camera

This workflow is useful when experiments need to be reproducible or when processing should be performed offline.

---

# 1. Clone the Repository

First clone the repository and move into it.

```bash
git clone https://github.com/<your-username>/human_model.git
cd human_model
```

All commands below assume you are inside the **repository root directory**.

---

# 2. Record a ZED Session to an SVO File

The ZED SDK provides a Python script that records the camera stream into an `.svo` or `.svo2` file.

## 2.1 Navigate to the recording script

```bash
cd recording/single\ camera/python
```

## 2.2 Start recording

Example command:

```bash
python3 recording.py --output_svo_file "../../../body tracking/python/recorded.svo2"
```

Explanation:

* `--output_svo_file` specifies where the recording will be saved
* `.svo` or `.svo2` are the only valid formats supported by the ZED recorder
* Quotes are not required here because the relative path avoids spaces in the command argument

The file will be saved inside:

```
human_model/body tracking/python/
```

---

## 2.3 Stop recording

Recording runs continuously.

Press:

```
CTRL + C
```

The SVO file will then be finalized and saved.

Example output file:

```
human_model/body tracking/python/recorded.svo2
```

---

# 3. Run the Human Detection Pipeline on the Recorded File

The project supports SVO playback through the argument:

```
--input_svo_file
```

When this argument is provided, the ZED SDK reads frames from the recorded file instead of the live camera.

---

## 3.1 Navigate to the project directory

From the repository root:

```bash
cd body\ tracking/python
```

---

## 3.2 Run detection on the recorded file

```bash
python3 main.py --input_svo_file recorded.svo2
```

Because the file is in the same directory, only the filename is required.

The system will:

1. Load the SVO file
2. Replay frames sequentially
3. Run ZED BODY_18 detection
4. Feed the detected skeleton to the inference pipeline
5. Display the viewer window

---

# 4. Optional Execution Modes

## 4.1 Run without visualization (faster)

```bash
python3 main.py --input_svo_file recorded.svo2 --no_view
```

This disables OpenGL and OpenCV visualization and allows faster processing.

---

## 4.2 Save processed video

```bash
python3 main.py --input_svo_file recorded.svo2 --output_video output.mp4
```

The system will save the processed frames to `output.mp4`.

---

## 4.3 Process SVO faster than real-time (optional code change)

Inside `ZEDBody18Stream.open()` add:

```python
init_params.svo_real_time_mode = False
```

This allows the pipeline to process frames as fast as the system permits rather than at camera FPS.

---

# 5. Complete Example Workflow

### Step 1 — Record

```bash
cd human_model
cd recording/single\ camera/python

python3 recording.py --output_svo_file "../../../body tracking/python/recorded.svo2"
```

Press **CTRL + C** to stop recording.

---

### Step 2 — Run detection

```bash
cd ../../..
cd body\ tracking/python

python3 main.py --input_svo_file recorded.svo2
```

---

# 6. Notes

* Only `.svo` and `.svo2` files are supported for playback.
* The BODY_18 skeleton detection is executed again during playback.
* Recording preserves the original camera frames so experiments can be repeated identically.

---

# 7. Troubleshooting

### Recording fails

Check that:

* The output directory exists
* The path is correct
* The path is quoted if it contains spaces

Example directory creation:

```bash
mkdir -p "body tracking/python"
```

---

### Verify recording file

Navigate to the directory containing the file:

```bash
cd body\ tracking/python
ls -lh recorded.svo2
```

The file size should increase during recording.

