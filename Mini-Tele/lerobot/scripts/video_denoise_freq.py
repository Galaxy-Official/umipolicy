import cv2
import numpy as np
import argparse
import sys

def create_filter_mask(shape, filter_type, cutoff_low, cutoff_high):
    """
    Creates a frequency domain filter mask.
    
    Args:
        shape: Tuple (height, width) of the image
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
        cutoff_low: Low cutoff frequency (normalized 0-0.5)
        cutoff_high: High cutoff frequency (normalized 0-0.5)
        
    Returns:
        2D filter mask (0s and 1s)
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2  # Center point
    
    # Create a distance map from center (normalized to [0, 0.5])
    y = np.linspace(-0.5, 0.5, rows)
    x = np.linspace(-0.5, 0.5, cols)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)  # Normalized distance from center [0, 0.707]
    
    # Create masks based on filter type
    mask = np.zeros((rows, cols), np.float32)
    
    if filter_type == 'lowpass':
        mask = (D <= cutoff_high).astype(np.float32)
    elif filter_type == 'highpass':
        mask = (D >= cutoff_low).astype(np.float32)
    elif filter_type == 'bandpass':
        mask = np.logical_and(D >= cutoff_low, D <= cutoff_high).astype(np.float32)
    elif filter_type == 'bandstop':
        mask = np.logical_or(D < cutoff_low, D > cutoff_high).astype(np.float32)
    
    return mask

def frequency_filter_frame(frame, filter_type, cutoff_low, cutoff_high):
    """
    Applies frequency domain filtering to a frame.
    
    Args:
        frame: Input BGR image
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
        cutoff_low: Low cutoff frequency (normalized 0-0.5)
        cutoff_high: High cutoff frequency (normalized 0-0.5)
        
    Returns:
        Filtered BGR image
    """
    # Convert to float32 and split channels
    b, g, r = cv2.split(frame.astype(np.float32))
    channels = [b, g, r]
    filtered_channels = []
    
    for channel in channels:
        # Fourier transform
        dft = cv2.dft(channel, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Create filter mask
        mask = create_filter_mask(channel.shape, filter_type, cutoff_low, cutoff_high)
        mask_3d = cv2.merge([mask, mask])  # For complex values
        
        # Apply filter
        filtered_shift = dft_shift * mask_3d
        
        # Inverse transform
        idft_shift = np.fft.ifftshift(filtered_shift)
        idft = cv2.idft(idft_shift, flags=cv2.DFT_REAL_OUTPUT)
        
        # Normalize and clip
        filtered_channel = np.clip(idft, 0, 255)
        filtered_channels.append(filtered_channel)
    
    # Merge channels and convert to uint8
    filtered = cv2.merge(filtered_channels)
    return np.clip(filtered, 0, 255).astype(np.uint8)

def laplacian_sharpen(frame, alpha=1.0, ksize=3):
    """
    Applies Laplacian sharpening to a frame.
    
    Args:
        frame: Input BGR image
        alpha: Sharpening strength
        ksize: Kernel size for Laplacian (1, 3, 5, or 7)
        
    Returns:
        Sharpened BGR image
    """
    if ksize not in [1, 3, 5, 7]:
        ksize = 3
    
    # Convert to float32 for processing
    frame_float = frame.astype(np.float32)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(frame_float, cv2.CV_32F, ksize=ksize)
    
    # Add Laplacian to original image
    sharpened = frame_float + alpha * laplacian
    
    # Clip and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255)
    return sharpened.astype(np.uint8)

def process_video(input_path, output_path, method, args):
    """
    Processes a video file frame by frame using frequency domain filtering.
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

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Subtract the first frame from the current frame to highlight changes
        changed_frame = cv2.addWeighted(frame, 1, first_frame, -0.4, 0)
        
        # Manage lightness
        changed_frame = cv2.addWeighted(changed_frame, 1, changed_frame, 0.5, 0)
        
        # Center crop the changed frame (keep central 2/3 portion)
        changed_frame = changed_frame[start_y:start_y+crop_height, 
                                    start_x:start_x+crop_width]

        # Apply frequency domain filtering
        filtered_frame = frequency_filter_frame(
            changed_frame, 
            method, 
            args.cutoff_low, 
            args.cutoff_high
        )

        # Apply optional sharpening to the filtered frame
        if args.sharpen:
            filtered_frame = laplacian_sharpen(
                filtered_frame, 
                alpha=args.alpha, 
                ksize=args.lap_ksize
            )
        
        # Saturation enhancement
        # Convert to HSV color space for saturation adjustment
        hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Increase saturation by 50% (adjust multiplier as needed)
        s = cv2.multiply(s, 0.65)
        s = np.clip(s, 0, 255).astype(np.uint8)  # Ensure values stay in valid range
        
        # Merge channels back and convert to BGR
        hsv = cv2.merge([h, s, v])
        saturated_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        out.write(saturated_frame)  # Write the saturation-enhanced frame
        frame_count += 1
        print(f"Processed frame {frame_count}", end='\r')

    cap.release()
    out.release()
    print(f"\nProcessed video saved to {output_path}")

if __name__ == "__main__":
    # Command-line arguments for VIDEO processing
    parser = argparse.ArgumentParser(
        description="Video processing with frequency domain filtering and optional sharpening"
    )
    parser.add_argument('method', choices=['lowpass', 'highpass', 'bandpass', 'bandstop'],
                        help="Frequency filter type to apply")
    parser.add_argument('input', help="Path to input MP4 file")
    parser.add_argument('output', help="Path to output MP4 file")
    parser.add_argument('--cutoff_low', type=float, default=0.05,
                        help="Low cutoff frequency (normalized 0-0.5)")
    parser.add_argument('--cutoff_high', type=float, default=0.2,
                        help="High cutoff frequency (normalized 0-0.5)")
    parser.add_argument('--sharpen', action='store_true',
                        help="Enable Laplacian sharpening after filtering")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Weight of Laplacian component in sharpening")
    parser.add_argument('--lap_ksize', type=int, default=3,
                        help="Kernel size for Laplacian operator (odd)")
    
    args = parser.parse_args()
    
    # Validate cutoff frequencies
    if args.cutoff_low < 0 or args.cutoff_low > 0.5:
        print("Error: cutoff_low must be between 0 and 0.5")
        sys.exit(1)
    if args.cutoff_high < 0 or args.cutoff_high > 0.5:
        print("Error: cutoff_high must be between 0 and 0.5")
        sys.exit(1)
    if args.cutoff_low >= args.cutoff_high:
        print("Error: cutoff_low must be less than cutoff_high")
        sys.exit(1)
    
    process_video(args.input, args.output, args.method, args)