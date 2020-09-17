import sys
import os
import numpy as np
import argparse
import pickle


parser = argparse.ArgumentParser(description='Taxonomic machine learning script',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser._action_groups.pop()
optional = parser.add_argument_group('optional arguments')

optional.add_argument('-p', choices=['FTC','F','T','C','FT','TC'], help='procedure: F - feature prediction, T - training, C - classifying uknown images',dest="procedure", default="FTC")
optional.add_argument('-a', choices=['SVM', 'DNN'], help='classifier type, support vector machine (SVM) or deep neural network (DNN)',dest="algorithm", default="SVM")
optional.add_argument('-b', metavar='1,2,3,4,5', help='CNN block to use, several may be selected for concatenation',dest="blocks", default="1,2,3,4,5")
optional.add_argument('-t', metavar='folder', help='folder with folders with images for training',dest="folder_to_train")
optional.add_argument('-c', metavar='folder', help='folder with images to classify',dest="folder_to_predict")
optional.add_argument('-f', metavar='folder', help='folder with extracted features',dest="feature_dest")
optional.add_argument('-m', metavar='folder', help='folder with saved models',dest="model_dest")
optional.add_argument('--resHeight', metavar='N', help='height of the images supplied to the CNN',dest="img_height", type=int, default=233)
optional.add_argument('--resWidth', metavar='N', help='width of the images supplied to the CNN',dest="img_width", type=int, default=312)
optional.add_argument('--batch', metavar='N', help='batch size',dest="batch_size", type=int, default=12)
optional.add_argument('--Nldnn', metavar='N', help='number of layers in the DNN, excluding the prediction layer, >=1',dest="nb_DNN_deep_layers", type=int, default=2)
optional.add_argument('--Nndnn', metavar='N', help='number of neurons in the DNN layers, excluding the prediction layer, >=1',dest="nb_DNN_neurons", type=int, default=320)
optional.add_argument('--Nepochs', metavar='N', help='number of epochs for DNN training',dest="nb_epochs", type=int, default=50)
optional.add_argument('--no-norm', dest='unnormalize', action='store_true', help='do not normalize features', default=False)
optional.add_argument('--Ksplits', metavar='N', help='number of cross-validation splits',dest="ksplits", type=int, default=10)

if len(sys.argv) == 1:
	parser.print_help()
	sys.exit()
else:
	args = parser.parse_args()
	procedure = vars(args)["procedure"]
	algorithm = vars(args)["algorithm"]
	folder_to_train = vars(args)["folder_to_train"]

	img_height = vars(args)["img_height"]
	img_width = vars(args)["img_width"]
	input_size = (img_height,img_width)
	nb_epochs = vars(args)["nb_epochs"]
	folder_to_predict = vars(args)["folder_to_predict"]
	feature_dest = vars(args)["feature_dest"]
	model_dest = vars(args)["model_dest"]

	batch_size = vars(args)["batch_size"]
	nb_DNN_deep_layers = vars(args)["nb_DNN_deep_layers"]
	nb_DNN_neurons = vars(args)["nb_DNN_neurons"]
	normalize = not vars(args)["unnormalize"]
	ksplits = vars(args)["ksplits"]

	blocks = set(vars(args)["blocks"].split(","))
	intblocks = [int(x) for x in blocks]

	#check option validity:
	if len(blocks) == 0 or min(intblocks) < 1 or max(intblocks) > 5:
		print ("incorrect blocks entered: "+vars(args)["blocks"]+", please use one or comma separated group of numbers from 1 to 5")
		sys.exit()
	if img_height == 0 or img_width == 0:
		print ("incorrect resolution entered: "+str(input_size)+", please use values greater than 0")
		sys.exit()
	if 'F' in procedure:
		if folder_to_train == None:
			print ("procedure has F selected, please use -t to provide the folder with training images")
			sys.exit()
	if 'T' in procedure:
		if feature_dest == None:
			if 'F' not in procedure:
				print ("procedure has T selected, please use -f to provide the folder with extracted features")
				sys.exit()
			else:
				print ("no folder for feature saving is provided, will use 'features' folder")
				feature_dest = "features"
		if model_dest == None:
			print ("no folder for model saving is provided, will use 'models' folder")
			model_dest = "models"
	if 'C' in procedure:
		if model_dest == None:
			if 'T' not in procedure:
				print ("procedure has C selected, please use -m to provide the folder with saved trained models")
				sys.exit()
			else:
				print ("no folder for model saving is provided, will use 'models' folder")
				model_dest = "models"
		if folder_to_predict == None:
			print ("procedure has C selected, please use -c to provide the folder with images to classify")
			sys.exit()

from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from keras.applications.vgg16 import VGG16


def mkdirfunc(dir1):
	if not os.path.exists (dir1):
		os.makedirs(dir1)

#### GET FEATURES ####
if 'F' in procedure:
	print ("####### feature extraction stage #######")

	model = VGG16(weights='imagenet', include_top=False)

	blocklist = []
	blocknums = []

	for bn in range(1,6):
		bs = str(bn)
		if bs in blocks:
			if bs == "1":
				bl = model.layers[-16].output 
			elif bs == "2":
				bl = model.layers[-13].output
			elif bs == "3":
				bl = model.layers[-9].output
			elif bs == "4":
				bl = model.layers[-5].output
			elif bs == "5":
				bl = model.layers[-1].output
			bl = GlobalAveragePooling2D()(bl)
			blocklist.append(bl)
			blocknums.append(bs)

	model = Model(inputs=model.input, outputs=blocklist)

	datagen = ImageDataGenerator(rescale=1./255.)
	generator = datagen.flow_from_directory(
		folder_to_train,
		target_size=(img_height, img_width),
		batch_size=batch_size,
		class_mode="sparse",
		shuffle=False)
	steps = generator.samples/batch_size
	X = model.predict_generator(generator,steps, verbose=1)
	Y = np.concatenate([generator.next()[1] for i in range(0, generator.samples, batch_size)])
	names = generator.filenames

	if len(blocklist) == 1:
		X = [X]

	mkdirfunc(feature_dest)
	for n, i in enumerate(X):
		with open(feature_dest+"/X-c"+str(blocknums[n])+".npy", 'wb') as f:
			np.save(f, i)
	with open(feature_dest+"/Y.npy", 'wb') as f:
		np.save(f, Y)
	with open(feature_dest+"/filenames.npy", 'wb') as f:
		np.save(f, names)
	with open(feature_dest+"/classes.txt", "w") as f:
		label_map = (generator.class_indices)
		for key, val in label_map.items():
			print(key, val, file=f)

############# TRAINING PART ###############

if "T" in procedure:
	print ("####### training stage #######")

	from sklearn.metrics import accuracy_score
	from sklearn.model_selection import StratifiedKFold
	mkdirfunc(model_dest)
	output_file = open(model_dest+"/evaluation_scores.csv", "w")
	output_dict = []

	labels_set = set()
	with open(feature_dest+"/classes.txt") as classes_handle:
		for line in classes_handle:
			line = line.strip().split()
			labels_set.add(line[0])

	if len(blocks) > 1:
		savedX = []
		for bn in range(1,6):
			bs = str(bn)
			if bs in blocks:
				savedX.append(np.load(feature_dest+"/X-c"+bs+".npy"))
		X = np.concatenate(savedX, 1)

	else:
		X = np.load(feature_dest+"/X-c"+next(iter(blocks))+".npy")

	Y = np.load(feature_dest +"/Y.npy")
	
	if normalize:
		X = np.sqrt(np.abs(X)) * np.sign(X)

	if algorithm == "SVM":

		from sklearn.svm import SVC, LinearSVC

		best_model = None
		best_score = None
		kfold = StratifiedKFold(n_splits=ksplits, shuffle=True, random_state=555)
		trial = 0
		for train, test in kfold.split(X, Y):
			clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr', max_iter=10000)
			clf.fit(X[train], Y[train])
			y_pred = clf.predict(X[test])
			acc = accuracy_score(Y[test],y_pred)
			print(acc)
			output_dict.append(acc)
			if trial == 0:
				best_model = clf
				best_score = acc
			else:
				if acc > best_score:
					best_model = clf
					best_score = acc
			trial += 1
		
		print ("best iteration accuracy:", best_score)
		print ("average accuracy:", sum(output_dict)/len(output_dict))

		pickle.dump(best_model, open(model_dest+"/SVMmodel.bin", 'wb'))
	
	elif algorithm == "DNN":

		kfold = StratifiedKFold(n_splits=ksplits, shuffle=True, random_state=555)
		trial = 0
		premodel = []
		for l in range(nb_DNN_deep_layers):
			premodel.append(Dense(nb_DNN_neurons, activation='relu', name='fc'+str(l)))
		premodel.append(Dense(len(labels_set), activation='softmax', name='predictions'))
		
		for train, test in kfold.split(X, Y):

			model = Sequential(premodel)

			model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

			steps = len(train)/batch_size
			history = model.fit(
				x = X[train],
				y= Y[train],
				steps_per_epoch = len(train) // batch_size,
				validation_data = (X[test], Y[test]),
				validation_steps = len(test) // batch_size,
				# validation_steps = 1,
				epochs = nb_epochs,
				verbose = 0)
			
			yhat_classes = np.argmax(model.predict(X[test]),axis=1)
			acc = accuracy_score(Y[test], yhat_classes)
			print(acc)
			output_dict.append(acc)
			if trial == 0:
				best_model = model
				best_score = acc
			else:
				if acc > best_score:
					best_model = model
					best_score = acc
			trial += 1

		print ("best iteration accuracy:", best_score)
		print ("average accuracy:", sum(output_dict)/len(output_dict))

		model_json = best_model.to_json()
		with open(model_dest+"/DNNmodel.json", "w") as json_file:
			json_file.write(model_json)
		best_model.save_weights(model_dest+"/DNNmodel.h5")

	for val1 in output_dict:
		print(val1, file=output_file)
	output_file.close()


if "C" in procedure:
	print ("####### classification stage #######")

	model = VGG16(weights='imagenet', include_top=False)

	blocklist = []
	blocknums = []

	for bn in range(1,6):
		if str(bn) in blocks:
			if str(bn) == "1":
				bl = model.layers[-16].output 
			elif str(bn) == "2":
				bl = model.layers[-13].output
			elif str(bn) == "3":
				bl = model.layers[-9].output
			elif str(bn) == "4":
				bl = model.layers[-5].output
			elif str(bn) == "5":
				bl = model.layers[-1].output
			bl = GlobalAveragePooling2D()(bl)
			blocklist.append(bl)
			blocknums.append(str(bn))

	model = Model(inputs=model.input, outputs=blocklist)

	labels_dict = {}
	with open(feature_dest+"/classes.txt") as classes_handle:
		for line in classes_handle:
			line = line.strip().split()
			labels_dict[line[1]] = line[0]
	from keras.preprocessing import image
	import glob
	evals = []
	evals.append("file,"+",".join(labels_dict.values())+",best_guess")

	if algorithm == "SVM":
		clf = pickle.load(open(model_dest+"/SVMmodel.bin", 'rb'))
	elif algorithm == "DNN":
		from keras.models import model_from_json
		json_file = open(model_dest+'/DNNmodel.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights(model_dest+"/DNNmodel.h5")
	
	for image_path in glob.glob(folder_to_predict+"/*.jpg"):
		img = image.load_img(image_path, target_size=input_size)
		x = image.img_to_array(img)
		x = x/255
		x = np.expand_dims(x, axis=0)
		images = np.vstack([x])

		extracted_features = model.predict(images)
		if len(blocks) > 1:
			features_to_predict = np.concatenate(extracted_features, 1)
		else:
			features_to_predict = extracted_features
		features_to_predict = np.sqrt(np.abs(features_to_predict)) * np.sign(features_to_predict)

		if algorithm == "SVM":
			predict_decision = clf.decision_function(features_to_predict[np.array([0])])
			predict_result = clf.predict(features_to_predict[np.array([0])])
			print(image_path, labels_dict[str(int(predict_result))])
			evals.append(image_path+","+",".join([str(z) for z in predict_decision.tolist()[0]])+","+labels_dict[str(int(predict_result))])	
		elif algorithm == "DNN":
			test1 = loaded_model.predict(features_to_predict)
			scores = test1.tolist()[0]
			idxs = (-test1).argsort()[:2].tolist()[0]
			print(image_path, labels_dict[str(idxs[0])])
			evals.append(image_path+","+",".join(str(z) for z in scores)+","+labels_dict[str(idxs[0])])

	with open(model_dest+"/predictions.csv", "w") as outf:
		for row in evals:
			print(row, file=outf)
			
