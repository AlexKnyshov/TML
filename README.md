# TML
Taxonomic Machine Learning

## Installation

Python 3 with the following packages is required:
Tensorflow (tested with 2.1.0)
Keras (tested with 2.3.1)
scikit-learn (tested with 0.22.2.post1)

<details>
<summary>Installation instructions</summary>
<p>

Tensorflow website has great instructions on how to install python and its packages and how to set up and use virtual environments.
Follow this link https://www.tensorflow.org/install/pip, follow the instructions for your operating system, use the method with virtual environments if possible. If python 3 is not installed, this website also has instructions on how to get it.
After tensorflow is installed, installing other packages using `pip` is straightforward. If using the virtual environments, make sure the other packages are installed under the same environment as tensorflow!
For keras the command would be as follows:
```
pip install --upgrade keras
```
or
```
pip3 install --upgrade keras
```

For installing scikit-learn, use the same environment and `pip`, if possible, like so:
```
pip install --upgrade scikit-learn
```
or
```
pip3 install --upgrade scikit-learn
```
For more information on scikit-learn installation, consult this page https://scikit-learn.org/stable/install.html.

Lastly, download or clone the repository with the script, or copy the script file from GitHub to your machine (`Code` button).

</p>
</details>

## Full workflow

The images used for training should be organized into a main folder with subfolders grouping images of the same class (taxon).

The images to classify (unknown taxa to predict ID of, or a test dataset to check the accuracy) should be grouped all in one folder.
Assuming the main folder is called `training_images`, while the images to classify are located in `images_to_classify` folder, the command would be as follows:

```
python train_and_predict.py -t ./training_images/ -c ./images_to_classify/
```

The results are saved in the `models` folder, the training accuracy is the `evaluation_scores.csv`, while the predictions of the images to classify are in the `predictions.csv`

## Only feature extraction and training

The training images are to be organized as in the `full workflow` section. To run the feature extraction and training, run:

```
python train_and_predict.py -t ./training_images/ -p FT
```

## Only image classification, using the trained network

For this stage only the model and the folder with images to classify are needed (no need for features folder, or the initial images used for training). To run the classifier, use the following command, assuming the images to classify are located in `images_to_classify` folder:

```
python train_and_predict.py -p C -c ./images_to_classify/
```

Make sure you use exactly same settings (if not default) that were used for the feature extraction and training, e.g., SVM or DNN, resolutions, configuration of blocks...