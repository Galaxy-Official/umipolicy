import cv2
import numpy as np
import argparse

# --- Frame/Image Processing Functions ---

def median_filter_frame(frame, ksize):
    """Applies a median blur to a single frame."""
    # The kernel size must be an odd number.
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(frame, ksize)

def opening_filter_frame(frame, kernel_size):
    """Applies a morphological opening to a single frame."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

def connectivity_filter_frame(frame, thresh, area_thresh, dilate_iter):
    """Removes small connected components (spots) from a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create a binary mask where spots are detected
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    
    # Find all connected components (i.e., contiguous regions of pixels)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    
    # Create a clean mask to store only the small spots we want to remove
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels): # Start from 1 to skip the background
        # If a component's area is below the threshold, add it to our removal mask
        if stats[i, cv2.CC_STAT_AREA] <= area_thresh:
            clean_mask[labels == i] = 255
            
    # Optionally dilate the mask to make the spots slightly larger,
    # which can help the inpainting process cover the edges better.
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        clean_mask = cv2.dilate(clean_mask, kernel, iterations=dilate_iter)
        
    # Use the mask of spots to inpaint the original frame, effectively removing them.
    return cv2.inpaint(frame, clean_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def laplacian_sharpen(frame, alpha=1.0, ksize=3):
    """Sharpens a frame using the Laplacian operator."""
    # The kernel size must be an odd number.
    if ksize % 2 == 0:
        ksize += 1
    lap = cv2.Laplacian(frame, cv2.CV_16S, ksize=ksize)
    lap = cv2.convertScaleAbs(lap)
    # Add the sharpened edges back to the original image
    sharpened = cv2.addWeighted(frame, 1.0, lap, alpha, 0)
    return sharpened

# --- Image Processing Function (for import) ---

def process_image(input_path, output_path, method, **kwargs):
    """
    Reads an image, applies denoising and optional sharpening, and saves the result.
    This function is intended to be imported into other scripts.

    Args:
        input_path (str): Path to the input JPG image.
        output_path (str): Path to save the output JPG image.
        method (str): Denoising method ('median', 'opening', 'connectivity').
        **kwargs: A dictionary of parameters for the filters.
                  Expected keys: ksize, thresh, area, dilate, sharpen,
                                 alpha, lap_ksize.
    """
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Cannot read image file: {input_path}")
        return

    # Apply the selected denoising method
    if method == 'median':
        denoised = median_filter_frame(frame, kwargs.get('ksize', 3))
    elif method == 'opening':
        denoised = opening_filter_frame(frame, kwargs.get('ksize', 3))
    elif method == 'connectivity':
        denoised = connectivity_filter_frame(
            frame,
            kwargs.get('thresh', 200),
            kwargs.get('area', 50),
            kwargs.get('dilate', 1)
        )
    else:
        print(f"Error: Unknown method '{method}'. Using original image.")
        denoised = frame

    # Apply optional sharpening
    if kwargs.get('sharpen', False):
        denoised = laplacian_sharpen(
            denoised,
            alpha=kwargs.get('alpha', 1.0),
            ksize=kwargs.get('lap_ksize', 3)
        )

    # Save the processed image
    cv2.imwrite(output_path, denoised)
    print(f"Processed image saved to {output_path}")


# --- Video Processing Function ---

def process_video(input_path, output_path, method, args):
    """
    Processes a video file frame by frame using command-line arguments.
    Subtracts the first frame from all subsequent frames before processing.
    Adds center crop to keep the central 2/3 portion of the image.
    Adds saturation enhancement at the final processing stage.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    # Read the first frame to use as a reference for subtraction
    ret, first_frame = cap.read()
    if not ret:
        raise IOError(f"Could not read the first frame from video: {input_path}")

    # Reset video to the beginning to process all frames in the loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Get video properties from the original frame
    height, width = first_frame.shape[:2]
    
    # Calculate center crop dimensions (2/3 of original size)
    crop_width = int(width * 2 / 3)
    crop_height = int(height * 2 / 3)
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Subtract the first frame from the current frame to highlight changes
        changed_frame = cv2.addWeighted(frame, 1, first_frame, -0.4, 0)
        
        # changed_frame = cv2.multiply(changed_frame, 0.6)  # fixed
        
        # manage lightness
        changed_frame = cv2.addWeighted(changed_frame, 1, changed_frame, 0.5, 0)
        
        # changed_frame = np.clip(changed_frame, 0, 180).astype(np.uint8)
        
        # Center crop the changed frame (keep central 2/3 portion)
        changed_frame = changed_frame[start_y:start_y+crop_height, 
                                    start_x:start_x+crop_width]

        # Denoise the resulting 'change' frame
        if method == 'median':
            denoised = median_filter_frame(changed_frame, args.ksize)
        elif method == 'opening':
            denoised = opening_filter_frame(changed_frame, args.ksize)
        else: # connectivity
            denoised = connectivity_filter_frame(
                changed_frame, args.thresh, args.area, args.dilate)

        # Apply optional sharpening to the denoised 'change' frame
        if args.sharpen:
            denoised = laplacian_sharpen(denoised, alpha=args.alpha, ksize=args.lap_ksize)
        
        # === SATURATION ENHANCEMENT ADDED HERE ===
        # Convert to HSV color space for saturation adjustment
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Increase saturation by 50% (adjust multiplier as needed)
        s = cv2.multiply(s, 0.65)
        s = np.clip(s, 0, 255).astype(np.uint8)  # Ensure values stay in valid range
        
        # Merge channels back and convert to BGR
        hsv = cv2.merge([h, s, v])
        saturated_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # === END SATURATION ENHANCEMENT ===

        out.write(saturated_frame)  # Write the saturation-enhanced frame

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")


if __name__ == "__main__":
    # This part handles command-line arguments for VIDEO processing.
    # The process_image function is intended to be imported, not run from CLI here.
    parser = argparse.ArgumentParser(description="Video denoising with optional Laplacian sharpening")
    parser.add_argument('method', choices=['median', 'opening', 'connectivity'],
                        help="Denoising method to apply")
    parser.add_argument('input', help="Path to input MP4 file")
    parser.add_argument('output', help="Path to output MP4 file")
    parser.add_argument('--ksize', type=int, default=3,
                        help="Kernel size (odd) for median/opening filters")
    parser.add_argument('--thresh', type=int, default=200,
                        help="Threshold for spot detection in connectivity method")
    parser.add_argument('--area', type=int, default=50,
                        help="Max area (px) of spots to remove")
    parser.add_argument('--dilate', type=int, default=1,
                        help="Dilation iterations before inpainting")
    parser.add_argument('--sharpen', action='store_true',
                        help="Enable Laplacian sharpening after denoising")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Weight of Laplacian component in sharpening")
    parser.add_argument('--lap-ksize', type=int, default=3,
                        help="Kernel size for Laplacian operator (odd)")
    args = parser.parse_args()

    process_video(args.input, args.output, args.method, args)

