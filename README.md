# Feature Prototype: Baseline CNN for Mammogram Classification

This is the feature prototype for the breast cancer detection project. It is not the
final system. Its job is to prove the hardest and least certain part of the project
works end to end: getting TensorFlow running on Apple Silicon, loading mammogram
images, training a convolutional network, and producing a prediction with an honest
evaluation. The classical models on the Wisconsin data are well understood and low
risk, so the convolutional network is the right thing to prototype now.

The script runs straight away on generated (synthetic) data, so you can confirm the
whole pipeline works and record a demo before the full dataset is downloaded. Once you
have the real images, you point it at a folder and nothing else changes.

---

## 1. Set up your M4 machine

Open Terminal and run these once.

```bash
# Xcode command line tools (needed for native builds)
xcode-select --install

# A clean virtual environment for the project
python3 -m venv ~/venv-bc
source ~/venv-bc/bin/activate
python -m pip install -U pip

# TensorFlow, then the Apple Metal plug-in for GPU acceleration
python -m pip install tensorflow
python -m pip install tensorflow-metal

# The rest of the prototype's dependencies
python -m pip install scikit-learn matplotlib numpy pillow
```

Every time you come back to work, reactivate the environment first:

```bash
source ~/venv-bc/bin/activate
```

If `tensorflow-metal` ever refuses to install, do not worry. It only adds GPU speed.
The prototype runs perfectly well on the CPU for a dataset this size, so you can skip
that one line and continue.

Check it worked:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

If a GPU device is listed, Metal is active. If the list is empty, you are on the CPU,
which is still fine.

---

## 2. Run it now on synthetic data

From the folder containing `prototype.py`:

```bash
python prototype.py
```

You will see the network build, train for ten epochs, and report its accuracy, AUC,
sensitivity, specificity and a confusion matrix on a held-out test set. It also saves
a trained model and two plots into an `outputs/` folder. This proves the pipeline runs.
Note that the synthetic accuracy will look very high, close to perfect, because the
generated images carry an easy signal on purpose. That is expected and only confirms
the wiring is correct. The real numbers in the next step will be lower and meaningful.

---

## 3. Run it on the real CBIS-DDSM images

Download the JPEG version of CBIS-DDSM from Kaggle, which is about 6 GB rather than the
163 GB original:

`https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset`

For a prototype you do not need all of it, and you must not sort it by hand. The folder
names under `jpeg/` are scan identifiers, not labels, so copying folders by position
would attach the wrong labels. The benign or malignant label for each image lives in the
CSV files. Use the included `prepare_data.py`, which reads those CSVs and sorts a sample
of images into the correct class folders for you. Point `--root` at the folder that
contains the `csv/` and `jpeg/` folders:

```bash
python prepare_data.py --root /path/to/CBIS-DDSM --out data --per_class 200
```

That creates the structure the prototype expects:

```
data/
  benign/
    ...
  malignant/
    ...
```

A few hundred images in each class is plenty to show the technique working. If the script
copies nothing, run it again and send me the diagnostic lines it prints, since the exact
column names in this dataset have shifted between versions and the matching is a one-line
adjustment.

Then run:

```bash
python prototype.py --data_dir data --epochs 15
```

The script reads the two folders, infers the labels from the folder names, splits the
images into training, validation and test sets, trains the baseline network, and prints
the same evaluation as before, this time on real mammograms.

Useful options: `--epochs` to train longer, `--img_size` to change the input resolution
(128 is the default and a sensible start), and `--n` to change the synthetic sample size.

---

## 4. What you get

Everything lands in `outputs/`:

- `baseline_cnn.keras`, the trained model.
- `training_curves.png`, accuracy and loss over the epochs for training and validation.
- `confusion_matrix.png`, the test-set confusion matrix.

The console also prints accuracy, AUC, sensitivity and specificity, which are the
figures to quote in your report and to read out in the demo.

---

## 5. Predict on a single image (image in, result out)

Once a model is saved in `outputs/`, `predict.py` runs it on one image and prints a
result. This is the smallest possible version of the final product, the image-in,
result-out core that your web interface will later wrap.

```bash
python predict.py --image path/to/mammogram.jpg
```

It prints benign or malignant with a confidence level. The full product arrives in the
development phase, when this same prediction step sits behind your Figma interface: the
screen sends an image to a small backend, the backend calls this model, and the result
returns with its explanation. So this script is not a throwaway, it is the seed the
product grows from.

---

## 6. Recording the demo video

The structure your tutor demonstrated works well, so follow the same three stages.

First, a sentence or two of context. One or two slides saying what the project is, why
the convolutional network is the part you chose to prototype, and that this is an
exploratory piece to prove the technique rather than the finished product.

Second, show it running. Run the script in the terminal and let the training scroll, then
open the two plots and the printed results, talking through the accuracy, the confusion
matrix and what sensitivity means here. Then run `predict.py` on a single image so the
viewer sees an actual prediction come out, which makes the whole thing concrete.

Third, show a little code. Open `prototype.py` and walk through two parts only, the data
loading that turns a folder of images into labelled batches, and the `build_model`
function that defines the network. You do not need to read every line.

Finally, evaluate and say what is next. Be honest about what works and what does not, and
say which problems you would fix and which would change the wider design. The draft below
gives you the words for this.

---

## 7. Draft evaluation for the preliminary report

The prototype set out to test the most uncertain part of the project, whether a
convolutional network for mammogram classification could be built and trained from
scratch on the available hardware, and on that question it succeeded. The environment
runs on the Apple Silicon machine, the pipeline loads images, trains a network and
returns a prediction, and the whole process is evaluated on a held-out test set rather
than on data the model has already seen. Proving this now removes the largest single risk
hanging over the project, since everything that follows depends on this pipeline existing.

The results should be read with care. On the generated data the network reaches near
perfect accuracy, which is expected and means only that the wiring is sound, because the
synthetic images carry a deliberately easy signal. The figures that matter are those from
the real mammograms, where a baseline network of this simplicity is expected to perform
modestly, somewhere well short of the published benchmarks, and that is the point of a
baseline. It establishes the floor that the later stages of the project are meant to
climb above. Reporting it honestly, including where it falls short, is more useful than
presenting an inflated number.

The shortcomings point directly at the next steps, and they are the steps already planned
in the design. A single small network will tend to overfit and to plateau at a modest
accuracy, so the next stage adds regularisation, and the stage after that introduces
transfer learning with VGG and EfficientNet, which is where the real gains are expected.
Because each stage is an extension of this working pipeline rather than a fresh start, a
weakness in the baseline does not threaten the project, it simply confirms the order of
work. If the network had failed to train at all, that would have forced a change to the
wider design, but it did not, so the design holds and development can proceed as planned.
