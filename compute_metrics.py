import argparse
from copy import deepcopy
from pprint import pprint

from utils.data_utils import *
from utils.get_accuracy import get_accuracies
from utils.helpers import extract_classifiers_batchSizes
from explanations import get_elbow_selections
from utils.helpers import get_computed_AI_selections

def main(args):

	explanation_dir = args.explanation_dir
	saved_models_path = args.saved_models_path
	dataset_base_path = args.dataset_dir
	result_path = args.result_file

	model_name,batch_size = extract_classifiers_batchSizes(args.classifiers_batchSizes)
	elbow_selections_path = args.elbow_selections

	channel_selection =  not (elbow_selections_path==None)
	print("performing channel selection") if channel_selection else print("performing time point selection")

	# otherwise load elbow selection, saliency maps and initial accuracies
	all_elbow_selections = np.load(elbow_selections_path, allow_pickle=True).item() if channel_selection else None

	# load dataset
	all_accuracies = {}
	for current_dataset in sorted(os.listdir(args.dataset_dir ) ):

		# load current dataset
		print("loading ", current_dataset,"...",end="\t")
		dataset_dir =  os.path.join(dataset_base_path,current_dataset)
		data = load_datasets(dataset_dir, current_dataset)
		print("Dataset loaded! \n Loading explanations...")

		XAI_results = np.load(os.path.join(explanation_dir, current_dataset+"_results.npz"),
								  allow_pickle=True)['results'].item()
		print("Explanations loaded!")

		# get elbow selections and AI's ones
		elbow_selections = get_elbow_selections(current_dataset,all_elbow_selections) if channel_selection else {}
		all_selections, init_accuracies = get_computed_AI_selections(XAI_results,{
			'ConvTran' : deepcopy(elbow_selections),
			'miniRocket': deepcopy(elbow_selections),
			'hydra' : deepcopy(elbow_selections)
		},{},"", channel_selection)


		# train models on selected dataset versions
		current_accuracies = get_accuracies(data,saved_models_path, all_selections, model_name,batch_size, init_accuracies,channel_selection)
		#pprint(current_accuracies ,indent=4)
		all_accuracies[current_dataset] = current_accuracies

		np.save( result_path ,all_accuracies)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("explanation_dir", type=str, help="dir where explanation results are stored")
	parser.add_argument("saved_models_path", type=str, help="folder where models will be saved")
	parser.add_argument("dataset_dir", type=str, help="directory where datasets are located.")
	parser.add_argument("result_file", type=str, help="file where to store new accuracies")
	parser.add_argument("--classifiers_batchSizes", nargs='+',help="classifier name is either hydra,"
																   "miniRocket or ConvTran")
	parser.add_argument("--elbow_selections", type=str,  nargs='?',default=None, help="optional argument."
		"file path where elbow selections are saved, implicitly defining whether channel selection (provided) """
			"or time point selection(not provided)")

	args = parser.parse_args()
	main(args)
