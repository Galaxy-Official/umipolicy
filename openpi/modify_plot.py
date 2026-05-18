import re

with open('/Users/macbookpro/Desktop/workspace/handcap/Postprocess/plot_pose_offset_compare.py', 'r') as f:
    code = f.read()

# Modify make_trajectory_video signature
code = code.replace(
    'def make_trajectory_video(\n        pose_mats_no_offset, pose_mats_with_offset, output_path, title,',
    'def make_trajectory_video(\n        pose_mats_tracker, pose_mats_eef, pose_mats_tcp, output_path, title,'
)
code = code.replace('no_offset_pos = pose_mats_no_offset[:, :3, 3]', 'tracker_pos = pose_mats_tracker[:, :3, 3]')
code = code.replace('with_offset_pos = pose_mats_with_offset[:, :3, 3]', 'eef_pos = pose_mats_eef[:, :3, 3]\n    tcp_pos = pose_mats_tcp[:, :3, 3]')
code = code.replace('all_pos = np.concatenate([no_offset_pos, with_offset_pos], axis=0)', 'all_pos = np.concatenate([tracker_pos, eef_pos, tcp_pos], axis=0)')
code = code.replace('deltas = with_offset_pos - no_offset_pos\n    offset_norm = np.linalg.norm(deltas, axis=1)', 'deltas = tcp_pos - eef_pos\n    offset_norm = np.linalg.norm(deltas, axis=1)')
code = code.replace('total = len(no_offset_pos)', 'total = len(tracker_pos)')

code = code.replace(
'''            ax.plot(no_offset_pos[:idx + 1, 0], no_offset_pos[:idx + 1, 1], no_offset_pos[:idx + 1, 2],
                    color="#ff7f0e", linewidth=3.0, label="without offset")
            ax.plot(with_offset_pos[:idx + 1, 0], with_offset_pos[:idx + 1, 1], with_offset_pos[:idx + 1, 2],
                    color="#1f77b4", linewidth=3.0, label="with offset")
            ax.plot(no_offset_pos[idx:, 0], no_offset_pos[idx:, 1], no_offset_pos[idx:, 2],
                    color="#ff7f0e", linewidth=1.0, alpha=0.18)
            ax.plot(with_offset_pos[idx:, 0], with_offset_pos[idx:, 1], with_offset_pos[idx:, 2],
                    color="#1f77b4", linewidth=1.0, alpha=0.18)''',
'''            ax.plot(tracker_pos[:idx + 1, 0], tracker_pos[:idx + 1, 1], tracker_pos[:idx + 1, 2],
                    color="#2ca02c", linewidth=2.0, label="Tracker")
            ax.plot(eef_pos[:idx + 1, 0], eef_pos[:idx + 1, 1], eef_pos[:idx + 1, 2],
                    color="#ff7f0e", linewidth=3.0, label="EEF")
            ax.plot(tcp_pos[:idx + 1, 0], tcp_pos[:idx + 1, 1], tcp_pos[:idx + 1, 2],
                    color="#1f77b4", linewidth=3.0, label="TCP")
            ax.plot(tracker_pos[idx:, 0], tracker_pos[idx:, 1], tracker_pos[idx:, 2],
                    color="#2ca02c", linewidth=1.0, alpha=0.15)
            ax.plot(eef_pos[idx:, 0], eef_pos[idx:, 1], eef_pos[idx:, 2],
                    color="#ff7f0e", linewidth=1.0, alpha=0.18)
            ax.plot(tcp_pos[idx:, 0], tcp_pos[idx:, 1], tcp_pos[idx:, 2],
                    color="#1f77b4", linewidth=1.0, alpha=0.18)'''
)

code = code.replace(
'''            p_no = no_offset_pos[idx]
            p_with = with_offset_pos[idx]
            delta = p_with - p_no
            ax.scatter(*p_no, color="#ff7f0e", s=80)
            ax.scatter(*p_with, color="#1f77b4", s=80)
            ax.quiver(
                p_no[0], p_no[1], p_no[2],
                delta[0], delta[1], delta[2],
                color="0.05", linewidth=2.6, arrow_length_ratio=0.13)
            draw_eef_orientation(
                ax, pose_mats_with_offset[idx], p_with, frame_len,
                orientation_frame, alpha=0.95, labels=True)''',
'''            p_track = tracker_pos[idx]
            p_eef = eef_pos[idx]
            p_tcp = tcp_pos[idx]
            delta = p_tcp - p_eef
            ax.scatter(*p_track, color="#2ca02c", s=60)
            ax.scatter(*p_eef, color="#ff7f0e", s=80)
            ax.scatter(*p_tcp, color="#1f77b4", s=80)
            ax.plot([p_track[0], p_eef[0]], [p_track[1], p_eef[1]], [p_track[2], p_eef[2]], color="0.4", linestyle="--")
            ax.quiver(
                p_eef[0], p_eef[1], p_eef[2],
                delta[0], delta[1], delta[2],
                color="0.05", linewidth=2.6, arrow_length_ratio=0.13)
            draw_eef_orientation(
                ax, pose_mats_tcp[idx], p_tcp, frame_len,
                orientation_frame, alpha=0.95, labels=True)'''
)

code = code.replace(
    'def make_three_view_videos(\n        pose_mats_no_offset, pose_mats_with_offset, base_output_path, title,',
    'def make_three_view_videos(\n        pose_mats_tracker, pose_mats_eef, pose_mats_tcp, base_output_path, title,'
)
code = code.replace(
'''        stats.append(make_trajectory_video(
            pose_mats_no_offset,
            pose_mats_with_offset,''',
'''        stats.append(make_trajectory_video(
            pose_mats_tracker,
            pose_mats_eef,
            pose_mats_tcp,'''
)

# Replace in main
code = code.replace(
'''    parser.add_argument(
        "--offset_mm",''',
'''    parser.add_argument(
        "--tcp_offset_mm",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 150.0],
        metavar=("X_MM", "Y_MM", "Z_MM"),
        help="Local TCP offset in mm.")
    parser.add_argument(
        "--tcp_euler_xyz",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("RX_DEG", "RY_DEG", "RZ_DEG"),
        help="Local TCP rotation in degrees (XYZ euler).")
    parser.add_argument(
        "--offset_mm",'''
)

code = code.replace(
'''    pose_mats_no_offset = np.stack([
        get_pose_transform_mats(
            htc_poses[idx], use_offset=False, offset_m=offset_m,
            offset_frame=args.offset_frame)
        for idx in pose_indices
    ])
    pose_mats_with_offset = np.stack([
        get_pose_transform_mats(
            htc_poses[idx], use_offset=True, offset_m=offset_m,
            offset_frame=args.offset_frame)
        for idx in pose_indices
    ])''',
'''    pose_mats_tracker = np.stack([
        get_pose_transform_mats(
            htc_poses[idx], use_offset=False, offset_m=offset_m,
            offset_frame=args.offset_frame)
        for idx in pose_indices
    ])
    pose_mats_eef = np.stack([
        get_pose_transform_mats(
            htc_poses[idx], use_offset=True, offset_m=offset_m,
            offset_frame=args.offset_frame)
        for idx in pose_indices
    ])
    
    tcp_offset_m = np.asarray(args.tcp_offset_mm, dtype=np.float64) / 1000.0
    tcp_offset_transform = np.eye(4)
    tcp_offset_transform[:3, 3] = tcp_offset_m
    tcp_rot = Rotation.from_euler('xyz', args.tcp_euler_xyz, degrees=True)
    tcp_offset_transform[:3, :3] = tcp_rot.as_matrix()

    pose_mats_tcp = pose_mats_eef @ tcp_offset_transform'''
)

code = code.replace(
'''    stats = plot_single_point_compare(
        pose_mats_no_offset, pose_mats_with_offset, point_idx, args.output, title,''',
'''    # Skip single point compare update for brevity, just use eef and tcp
    stats = plot_single_point_compare(
        pose_mats_eef, pose_mats_tcp, point_idx, args.output, title,'''
)

code = code.replace(
'''    if args.video_output is not None:
        video_stats = make_three_view_videos(
            pose_mats_no_offset,
            pose_mats_with_offset,''',
'''    if args.video_output is not None:
        video_stats = make_three_view_videos(
            pose_mats_tracker,
            pose_mats_eef,
            pose_mats_tcp,'''
)

with open('/Users/macbookpro/Desktop/workspace/handcap/Postprocess/plot_three_trajectories.py', 'w') as f:
    f.write(code)
print("Done writing plot_three_trajectories.py")
