# 📁 Data Overview: Real vs Synthetic in AI_VolumeNet

AI_VolumeNet uses two distinct types of image data to build and validate its volume estimation pipeline:

---

## 🟢 Real Data (`data/real/`)

These are images captured from actual cameras (e.g., webcam, phone, DSLR) in uncontrolled environments.

- **Purpose:** Used for testing and validating the pipeline in real-world conditions.
- **Characteristics:**
  - Natural lighting, shadows, and occlusions
  - No guaranteed ground truth for object dimensions
  - Used for qualitative evaluation and plausibility checks
- **Example Use:** Webcam demo, YCB object test, generalization study

---

## 🔵 Synthetic Data (`data/synthetic/`)

These are programmatically generated images with known object dimensions and camera parameters.

- **Purpose:** Used for training and quantitative evaluation of the dimension and volume estimation models.
- **Characteristics:**
  - Perfect ground truth for bounding boxes and (L, W, H)
  - Controlled camera intrinsics and object scale
  - Enables supervised learning and precise error metrics
- **Example Use:** Training MLP/CNN for dimension estimation, computing volume error (MAE, RMSE)

---

## 🧠 Why Both Are Needed

| Purpose               | Synthetic Data            | Real Data                |
|-----------------------|---------------------------|--------------------------|
| Train dimension model | ✅ Ground truth available   | ❌ No labels              |
| Evaluate metrics      | ✅ Precise volume error     | ❌ Only qualitative       |
| Test generalization   | ❌ Too clean                | ✅ Real-world complexity  |
| Demo plausibility     | ❌ Not convincing visually  | ✅ Looks realistic         |

---

## 🔄 Data Flow in the Pipeline

```text
Synthetic Data (images + dimensions.csv)
        ↓
Train Dimension Estimator (MLP/CNN)
        ↓
Evaluate Volume Metrics (MAE, RMSE, error plots)
        ↓
Real Data (images only)
        ↓
Test Generalization & Demo Plausibility
