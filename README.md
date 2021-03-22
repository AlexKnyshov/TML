# Taxonomic Machine Learning

## References

**The script was published in:**
Alexander Knyshov, Samantha Hoang, Christiane Weirauch, Pretrained Convolutional Neural Networks Perform Well in a Challenging Test Case: Identification of Plant Bugs (Hemiptera: Miridae) Using a Small Number of Training Images, Insect Systematics and Diversity, Volume 5, Issue 2, March 2021, 3, https://doi.org/10.1093/isd/ixab004

The algorithm is in part based on:
Miroslav Valan, Karoly Makonyi, Atsuto Maki, Dominik Vondráček, Fredrik Ronquist, Automated Taxonomic Identification of Insects with Expert-Level Accuracy Using Effective Feature Transfer from Convolutional Networks, Systematic Biology, Volume 68, Issue 6, November 2019, Pages 876–895, https://doi.org/10.1093/sysbio/syz014

## Installation

Python 3 (64 bit version) with the following packages is required:
* Tensorflow (tested with 2.0.0, 2.1.0, and 2.3.0)
* Keras (tested with 2.3.1, 2.4.3)
* scikit-learn (tested with 0.22, 0.23)
* Pillow (tested with 7.2)

Should work on Windows, Mac, and Linux; was tested on Win 10 and Ubuntu 18.04.

<details>
<summary><b>Installation instructions</b></summary>
<p>

Tensorflow website has great instructions on how to install python and its packages and how to set up and use virtual environments.
Follow this link https://www.tensorflow.org/install/pip, follow the instructions for your operating system, use the method with virtual environments if possible. If python 3 is not installed, this website also has instructions on how to get it. **Make sure to get 64bit version!**

After tensorflow is installed, installing other packages using `pip` is straightforward. If using the virtual environments, make sure the other packages are installed under the same environment as tensorflow!
For keras the command would be as follows:
```
pip install --upgrade keras
```

For scikit-learn:
```
pip install --upgrade scikit-learn
```

for Pillow:
```
pip install --upgrade pillow
```

Other associated python packages should be installed automatically.

Lastly, download or clone the repository with the script (`Code` button), or copy the script file from GitHub to your machine (click on the `train_and_predict.py` file, then click on `raw` button above the script text, then save the file as `train_and_predict.py` in a folder on your machine that you intend to work in.

</p>
</details>

## Full workflow

The images used for training should be organized into a main folder with subfolders grouping images of the same class (taxon).
**Supported image formats in Keras are: jpeg, png, bmp, gif.**
The images to classify (unknown taxa to predict ID of, or a test dataset to check the accuracy) should be grouped all in one folder.
```
-- training_images
      |
      |-- taxon1
      |    |
      |    |-- image1
      |    |-- image2
      |    |-- image3
      |
      |-- taxon2
      |    |
      |    |-- image1
      |    |-- image2
      |    |-- image3
      |
      |-- taxon3
           |
           |-- image1
           |-- image2
           |-- image3

-- images_to_classify
      |
      |-- image1
      |-- image2
      |-- image3
```
Assuming Python 3 executable is called `python`, **the script and all the folders are located in the working directory**, the main folder is called `training_images`, while the images to classify are located in `images_to_classify` folder, the command would be as follows:

```
python train_and_predict.py -t training_images -c images_to_classify
```

During the run of the script, especially at the beginning, lots of service messages and warnings might pop up, that is normal. Additionally, at the first run on the system or virtual environment, the script will download the VGG16 model, which might take some time.

The results are saved in the `models` folder, the training accuracy is the `evaluation_scores.txt`, while the predictions of the images to classify are in the `predictions.csv`

**If too much memory is used, reduce batch size** (down to 1 for the worst possible scenario)

Execute `python train_and_predict.py` command to get the help on other options available.

## Only feature extraction and training

The training images are to be organized as in the `full workflow` section. To run the feature extraction and training, run:

```
python train_and_predict.py -t training_images -p FT
```

## Only image classification, using the trained network

For this stage only the model and the folder with images to classify are needed (no need for features folder, or the initial images used for training). To run the classifier, use the following command, assuming the images to classify are located in `images_to_classify` folder and the model files are stored in `models` folder:

```
python train_and_predict.py -p C -c images_to_classify -m models
```

Make sure you use exactly same settings (if not default) that were used for the feature extraction and training, e.g., SVM or DNN, resolutions, configuration of blocks...
