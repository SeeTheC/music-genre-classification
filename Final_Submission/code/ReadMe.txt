[READ ME]
Project: Music Genere Classcification
-------------------------------------------------
*Below is project structure:

1. PreProcessing.ipynb [File]: Used for extracting features from the audio file. It extract 20 features from the audio files. The dataset link is http://marsyasweb.appspot.com/download/data_sets/ . This dataset is having 100 audio file per gener and there are 10 gener. It generates the
"train.csv" and "test.csv" files.

2. neural_network_multilclass.ipynb [File]: It uses the "data/train.csv" for the training the nn and "test.csv" for predicting it.

3. DecisionTree.ipynb [File]: Code of decision tree written by us. It uses the "data/train.csv" for creating the decision tree  and "test.csv" for predicting it.
4. RandomForest.ipynb [File]: Code of random forest written by us. It uses the "data/train.csv" for creating the decision tree  and "test.csv" for predicting it.

5. CNN.ipynb [File]:Code of Convolutional Neural Network. It uses powerSpectrograms200 as input dataset for predicting the genres.

6. Training_with_sklearn.ipynb [File]: Used sklearn algo for training the models and gives the status of various algo.
 
Note: Files 1 to 6 are "jupyter notebook" files.

7. weights.txt [File] : Trained weights for the neural network.
8. data [Folder]: It contains the features files extracted from the audio files.
9. python_format [Folder]: Python code format code for 1 to 6 files, can be run through shell using python3.0 command
10. log/images [Folder]: Containes the images of the decision after the creation of tree.
-------------------------------------------------
*How Run the code.
Method 1: From Juypter
1. Install Jupyter Notebook and "librosa" lib.
2. Download the audio dataset from above given url and extract that dataset.
3. Open the "1_PreProcessing.ipynb" and give the path of exctracted folder.
4. Run the "1_PreProcessing.ipynb" by "shift enter"
5. Code will generated 3 files "audiofeatures.csv", "train.csv", and "test.csv".
6. One can train the system using remaining 5 files "neural_network_multilclass.ipynb","DecisionTree.ipynb", "RandomForest.ipynb", "CNN.ipynb ", and "Training_with_sklearn.ipynb".

Method 2: From Shell.
1. Open sheel and install "librosa" lib.
2. Download the audio dataset from above given url and extract that dataset.
3. Open the "1_PreProcessing.py" and give the path of exctracted folder.
4. Run the "1_PreProcessing.py" from shell "python 1_PreProcessing.py"
5. Code will generated 3 files "audiofeatures.csv", "train.csv", and "test.csv".
6. One can train the system using remaining 4 files "neural_network_multilclass.py","DecisionTree.py", "RandomForest.py", and "Training_with_sklearn.py".

-------------------------------------------------
*Saved model files:
1. weights.txt containes trained weigth for neural network with 7 hidden layers and 40 nodes per hidden layers.
2. There is no trained model file for DecisionTree and Random Forest as we cann't these train tress on to the files.

