import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
import argparse


def compute_trajectory_displacement(trajectory):
    """
    Compute total displacement of a trajectory
    """
    displacement = 0.0
    for t in range(len(trajectory) - 1):
        displacement += np.linalg.norm(trajectory[t+1] - trajectory[t])
    return displacement


def extract_action_directions(trajectories):
    """Extract action directions from trajectories
    
    Args:
        trajectories: (N, T, 3) array of N trajectories with T timesteps
        
    Returns:
        action_positions: (N, T-1, 3) positions where actions are taken
        action_directions: (N, T-1, 3) normalized action direction vectors
    """
    N, T, _ = trajectories.shape
    
    action_positions = []
    action_directions = []
    
    for i in range(N):
        traj = trajectories[i]  # (T, 3)
        for t in range(T - 1):
            pos = traj[t]
            direction = traj[t+1] - traj[t]  # Displacement vector
            
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
                action_positions.append(pos)
                action_directions.append(direction)
    
    return np.array(action_positions), np.array(action_directions)


def fit_revolute_axis_direction(action_directions):
    """Fit revolute joint axis direction using the common action plane method
    
    The axis direction n should minimize:
        sum |n · a_t^dir|
    which means n should be perpendicular to all action directions.
    
    This is equivalent to finding the eigenvector with smallest eigenvalue
    of the matrix A^T A, where A is the matrix of action directions.
    
    Args:
        action_directions: (M, 3) array of normalized action direction vectors
        
    Returns:
        axis_direction: (3,) normalized axis direction vector
    """
    # Compute covariance matrix of action directions
    # The axis should be perpendicular to most directions
    cov = np.dot(action_directions.T, action_directions)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # The axis direction is the eigenvector with smallest eigenvalue
    # (most perpendicular to action directions)
    axis_direction = eigenvectors[:, 0]
    
    # Ensure consistent orientation (pointing up in z-direction)
    if axis_direction[2] < 0:
        axis_direction = -axis_direction
    
    return axis_direction


def compute_closest_point_between_lines(p1, d1, p2, d2):
    """Compute closest point (midpoint) between two lines in 3D
    
    Args:
        p1, d1: point and direction of line 1
        p2, d2: point and direction of line 2
        
    Returns:
        midpoint: (3,) midpoint between closest points
    """
    w = p1 - p2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w)
    e = np.dot(d2, w)
    
    denom = a * c - b * b
    if abs(denom) < 1e-8:
        return (p1 + p2) / 2
    
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    
    pt1 = p1 + t * d1
    pt2 = p2 + s * d2
    
    return (pt1 + pt2) / 2


def fit_axis_position_from_single_trajectory(trajectory, axis_direction):
    """Fit axis position from a single trajectory
    
    For one trajectory with T timesteps:
    1. Extract T-1 actions (displacement vectors)
    2. Construct T-1 lines perpendicular to actions
    3. Compute all pairwise intersections (C(T-1, 2) pairs)
    4. Return mean of intersection points
    
    Args:
        trajectory: (T, 3) single trajectory
        axis_direction: (3,) axis direction
        
    Returns:
        axis_candidate: (3,) this trajectory's estimate of axis position
    """
    T = len(trajectory)
    
    actions = []
    for t in range(T - 1):
        pos = trajectory[t]
        direction = trajectory[t+1] - trajectory[t]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
            # Construct line perpendicular to action and axis
            line_dir = np.cross(direction, axis_direction)
            line_norm = np.linalg.norm(line_dir)
            if line_norm > 1e-8:
                line_dir = line_dir / line_norm
                actions.append({
                    'pos': pos,
                    'line_dir': line_dir
                })

    # Compute all pairwise intersections within this trajectory
    intersections = []
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            p1 = actions[i]['pos']
            d1 = actions[i]['line_dir']
            p2 = actions[j]['pos']
            d2 = actions[j]['line_dir']
            
            intersection = compute_closest_point_between_lines(p1, d1, p2, d2)
            intersections.append(intersection)
    
    # Return mean of all intersections from this trajectory
    return np.mean(intersections, axis=0)


def fit_revolute_axis_position(trajectories, axis_direction):
    """Fit revolute joint axis position using per-trajectory voting
    
    Correct Logic:
    1. For each trajectory independently:
       - Extract actions (T-1 from T timesteps)
       - Compute pairwise line intersections
       - Get one axis candidate point
    2. Average all 256 candidate points
    
    Args:
        trajectories: (N, T, 3) array of N trajectories
        axis_direction: (3,) normalized axis direction
        
    Returns:
        axis_position: (3,) a point on the rotation axis
    """
    N = len(trajectories)
    
    print(f"    Processing {N} trajectories independently...")
    
    # Get one axis candidate from each trajectory
    axis_candidates = []
    for i in range(N):
        candidate = fit_axis_position_from_single_trajectory(
            trajectories[i], axis_direction
        )
        axis_candidates.append(candidate)
    
    axis_candidates = np.array(axis_candidates)  # (N, 3)
    
    axis_position = np.mean(axis_candidates, axis=0)
    
    print(f"Computed axis position from {len(axis_candidates)} trajectory candidates")
    
    return axis_position


def fit_rotation_axis_from_trajectories(trajectories):
    """Fit rotation axis and anchor point from trajectories
    
    Correct method:
    1. Extract all actions to find axis direction (global)
    2. For each trajectory, find axis position independently
    3. Average all trajectory estimates
    
    Args:
        trajectories: (N, T, 3) array of trajectories
        
    Returns:
        anchor_point: (3,) point on the rotation axis
        axis_direction: (3,) normalized axis direction
        axis_point2: (3,) second point on the axis
    """
    # Step 1: Extract all actions to find axis direction
    action_positions, action_directions = extract_action_directions(trajectories)
    print(f"  Extracted {len(action_directions)} actions from {len(trajectories)} trajectories")
    
    # Step 2: Fit axis direction (global, from all actions)
    axis_direction = fit_revolute_axis_direction(action_directions)
    print(f"  Fitted axis direction: {axis_direction}")
    
    # Step 3: Fit axis position (per-trajectory voting)
    axis_position = fit_revolute_axis_position(trajectories, axis_direction)
    print(f"  Fitted axis position: {axis_position}")
    
    # The anchor point is on the axis
    anchor_point = axis_position
    
    # Second point on axis for representation
    axis_point2 = anchor_point + axis_direction
    
    return anchor_point, axis_direction, axis_point2


def compute_part_point(trajectories, anchor_point, axis_direction):
    """Compute part point as the start point farthest from the axis
    
    Args:
        trajectories: (N, T, 3) array
        anchor_point: (3,) point on axis
        axis_direction: (3,) axis direction
        
    Returns:
        part_point: (3,) the part point
    """
    start_points = trajectories[:, 0, :]  # (N, 3)
    
    # Compute distance from each start point to the axis
    distances = []
    for pt in start_points:
        vec = pt - anchor_point
        proj = np.dot(vec, axis_direction) * axis_direction
        perp = vec - proj
        dist = np.linalg.norm(perp)
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Select the farthest point
    farthest_idx = np.argmax(distances)
    part_point = start_points[farthest_idx]
    
    return part_point


def process_one_result(result_path, output_dir):
    """Process one inference result file
    
    Note: All 3D coordinates in the input are already normalized using point cloud parameters.
    
    Args:
        result_path: path to inference result pkl file
        output_dir: directory to save processed results
        
    Returns:
        processed_result: dict with fitted axis and anchor
    """
    # Load inference result
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
    
    data_name = result['data_name']
    trajectories_all = result['trajectories']  # List of (Q, T, 3) arrays - already normalized
    norm_center = result['norm_center']  # (3,) normalization center from point cloud
    norm_scale = result['norm_scale']    # scalar normalization scale from point cloud
    
    print(f"\nProcessing {data_name}")
    print(f"  Number of grippers: {len(trajectories_all)}")
    print(f"  Normalization: center={norm_center}, scale={norm_scale:.4f}")
    
    # Concatenate all trajectories from all grippers
    all_trajs = []
    for traj_group in trajectories_all:
        # traj_group is (Q, T, 3), reshape to (Q, T, 3)
        for i in range(len(traj_group)):
            all_trajs.append(traj_group[i])  # Each is (T, 3)
    all_trajs = np.array(all_trajs)  # (N_total, T, 3)
    
    # Compute displacement for each trajectory
    displacements = []
    for traj in all_trajs:
        disp = compute_trajectory_displacement(traj)
        displacements.append(disp)
    displacements = np.array(displacements)
    
    print(f"  Total trajectories: {len(all_trajs)}")
    print(f"  Displacement range: [{displacements.min():.4f}, {displacements.max():.4f}]")
    
    # Select top trajectories with largest displacement (up to 5*256 = 1280)
    num_to_select = min(5*256, len(all_trajs))
    top_indices = np.argsort(displacements)[-num_to_select:]
    selected_trajs = all_trajs[top_indices]  # (num_to_select, T, 3)
    
    print(f"  Selected {len(selected_trajs)} trajectories")
    print(f"  Selected displacement range: [{displacements[top_indices].min():.4f}, {displacements[top_indices].max():.4f}]")
    
    # Fit rotation axis
    anchor_point, axis_direction, axis_point2 = fit_rotation_axis_from_trajectories(selected_trajs)
    
    # Compute part point
    part_point = compute_part_point(selected_trajs, anchor_point, axis_direction)
    print(f"  Part point: {part_point}")
    
    processed_result = {
        'data_name': data_name,
        'pred_anchor_points': anchor_point,        # normalized coordinates
        'pred_joint_directions': axis_direction,   # direction vector (unit vector)
        'pred_part_points': part_point,            # normalized coordinates
        'norm_center': norm_center,                # normalization from point cloud
        'norm_scale': norm_scale,                  # normalization from point cloud
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'results_{data_name}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(processed_result, f)
    
    print(f"  Saved to: {output_path}")
    
    return processed_result


def main():
    parser = argparse.ArgumentParser('Laptop Postprocessing - Fit Rotation Axis')
    parser.add_argument('--infer_dir', type=str,
                        default='output/laptop_trajectories',
                        help='Directory containing inference results')
    parser.add_argument('--output_dir', type=str,
                        default='output/laptop_processed',
                        help='Output directory for processed results')
    
    args = parser.parse_args()
    
    result_files = glob.glob(os.path.join(args.infer_dir, '*.pkl'))
    result_files = sorted(result_files)
    
    if len(result_files) == 0:
        print(f"No result files found in {args.infer_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    success_count = 0
    error_count = 0
    
    for result_path in tqdm(result_files, desc="Processing"):
        try:
            processed_result = process_one_result(result_path, args.output_dir)
            success_count += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to process {result_path}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    print("\n" + "="*70)
    print("Postprocessing Complete")
    print("="*70)
    print(f"Total files: {len(result_files)}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Processed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

