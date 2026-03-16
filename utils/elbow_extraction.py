import numpy as np

def _detect_knee_point(values: list[float], indices: list[int]) -> list[int]:
	"""Find elbow point."""
	n_points = len(values)
	all_coords = np.vstack((range(n_points), values)).T
	first_point = all_coords[0]
	line_vec = all_coords[-1] - all_coords[0]
	line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
	vec_from_first = all_coords - first_point
	scalar_prod = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
	vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
	vec_to_line = vec_from_first - vec_from_first_parallel
	dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
	knee_idx = np.argmax(dist_to_line)
	knee = values[knee_idx]
	best_dims = [idx for (elem, idx) in zip(values, indices) if elem > knee]
	if len(best_dims) == 0:
		# return all dimensions if no elbow point is found
		return indices
	return best_dims

