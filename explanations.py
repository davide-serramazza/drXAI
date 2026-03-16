import numpy as np

from utils.elbow_extraction import _detect_knee_point

####################### functions to extract and order time points selection ##############
def extract_timePoints_features_names(attribution):
	"""
	Extracts feature relevance values and corresponding feature names from the input
	attribution tensor.

	:param attribution: A 3D numpy array representing the saliency map in the format
	(samples, channels, timepoints).

	:return:
		- feature_relevance: A 1D numpy array containing averaged feature attribution
		 across samples
		- feature_names: A list of strings each of those indicating start and end
		position of the corresponding feature.
	"""

	# identify where the unique values of the saliency maps are
	all_intervals = np.where(np.diff(attribution)!=0)

	# finding the index i.e. the training sample having more unique values
	unique_values, counts = np.unique(all_intervals[0], return_counts=True)
	max_count_index = np.argmax(counts)


	# take its intervals
	intervals = np.take(all_intervals[-1],	np.where(all_intervals[0]==max_count_index))

	# add last element at the end as rep. of the last feature
	to_gather = np.append( intervals[0],  attribution.shape[-1] -1 )

	# gather these attributions
	all_feature_relevance = np.take_along_axis(  attribution[:,0,:],
												 np.expand_dims(to_gather,axis=0),axis=-1)
	feature_relevance = np.average(all_feature_relevance,axis=0)

	# feature names as startingTimePoint:EndTimePoint
	intervals = ["0"] + to_gather.astype(str).tolist()
	feature_names = [ intervals[i]+":"+intervals[i+1] for i in range( len(intervals)-1 )]

	return feature_relevance, feature_names


def order_timePoints_features_names(feature_relevance, feature_names):
	"""
	Order time points and relative feature names in descending order.

	:param feature_relevance:	vector of features relevance values
	:param feature_names:		vector of corresponding features names

	:return:
		ordered_relevance:		ordered vector of features relevance values
		ordered_idx:			ordered vector of corresponding features names
	"""

	# identify the argsort order
	order = np.flip(np.argsort(feature_relevance))
	# apply the order, i.e. sort, to both feature relevance and feature names
	ordered_relevance = feature_relevance[order].tolist()
	ordered_idx =np.array(feature_names)[order].tolist()

	return ordered_relevance, ordered_idx
	

##################### functions to select features ###############################

def extract_selection_absFirst(attribution, channels=False):
	"""
	function to extract relevant time points/channels out of a saliency maps
	:param attribution:		the saliency map where to extract relevant attribution
	:return: 				selected channels
	"""

	# define extract and sort functions according to time points vs channels selection
	extract_features_names = (
		lambda x : (np.average(np.average(np.abs(x) , axis=-1),axis=0),None)
			) if channels else (
		lambda x :extract_timePoints_features_names(np.abs(x))
	)

	order_features_names = (lambda x,y : ( np.flip(np.sort(x).tolist()) , np.flip(np.argsort(x)).tolist() )) \
		if channels else order_timePoints_features_names	# ignoring the second argument in case of channels selection

	# apply above defined functions
	feature_relevance,feature_names = extract_features_names(attribution)
	ordered_relevance , ordered_idx = order_features_names(feature_relevance,feature_names)

	# knee cut
	knee_selection = _detect_knee_point(ordered_relevance,ordered_idx )

	return knee_selection


def extract_selection_avgFirst(attribution, channels=False):
	"""
	function to extract relevant time points/channels out of a saliency maps
	:param attribution:		the saliency map where to extract relevant attribution
	:return: 				selected channels
	"""

	# define extract and sort functions according to time points vs channels selection
	extract_features_names = (lambda x : (np.average(np.average(x , axis=-1),axis=0),None) ) \
		if channels else extract_timePoints_features_names

	order_features_names = (lambda x,y : ( np.flip(np.sort(x).tolist()) , np.flip(np.argsort(x)).tolist() )) \
		if channels else order_timePoints_features_names

	# apply function to extract feature relevance defined above
	feature_relevance,feature_names = extract_features_names(attribution)
	feature_relevance =  np.abs(feature_relevance)

	ordered_relevance , ordered_idx = order_features_names(feature_relevance, feature_names)
	knee_selection = _detect_knee_point(ordered_relevance,ordered_idx )

	return knee_selection





################################## other functions ##############################################

def get_elbow_selections(current_data,elbows):
	return {
		'elbow_pairwise' : 	{'selection' :
								   elbows[current_data]['Pairwise'] },
		'elbow_sum' :  		{'selection' :
								  elbows[current_data]['Sum']}
	}



