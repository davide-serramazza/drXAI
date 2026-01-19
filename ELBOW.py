from aeon.transformations.collection.channel_selection import ElbowClassSum, ElbowClassPairwise

def get_ELBOW_scores(ranks, dataset, X_train, y_train):
	# get elbow cut #
	ecp = ElbowClassPairwise()
	ecp.fit(X_train, y_train)
	current_rank = ecp.distance_frame.sum(axis=1).sort_values(ascending=False)
	ranks[dataset] = {"absolute" : current_rank , "relative" : current_rank/current_rank.max() }