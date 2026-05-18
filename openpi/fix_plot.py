import re

with open('/Users/macbookpro/Desktop/workspace/handcap/Postprocess/plot_three_trajectories.py', 'r') as f:
    code = f.read()

# Just comment out the single point compare logic since the user wants a video.
code = code.replace(
'''    stats = plot_single_point_compare(
        pose_mats_eef, pose_mats_tcp, point_idx, args.output, title,
        orientation_frame=args.orientation_frame)
    print(f"Saved: {Path(args.output).expanduser().resolve()}")
    print(stats)''',
'''    stats = "Skipped single point"'''
)

with open('/Users/macbookpro/Desktop/workspace/handcap/Postprocess/plot_three_trajectories.py', 'w') as f:
    f.write(code)
print("Fixed")
