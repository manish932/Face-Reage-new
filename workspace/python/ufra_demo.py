#!/usr/bin/env python3
"""
UFRa Python Demo - Universal Face Re-Aging Demo Script
Demonstrates the capabilities of the UFRa library for face aging and de-aging.
"""

import cv2
import numpy as np
import argparse
import sys
import os

try:
    import pyufra
except ImportError:
    print("Error: pyufra module not found. Please build and install the Python bindings.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='UFRa Python Demo')
    parser.add_argument('--input', '-i', required=True, help='Input image or video file')
    parser.add_argument('--output', '-o', required=True, help='Output image or video file')
    parser.add_argument('--age', '-a', type=float, default=30.0, help='Target age (0-100)')
    parser.add_argument('--mode', '-m', choices=['feedforward', 'diffusion', 'hybrid', 'auto'], 
                        default='feedforward', help='Processing mode')
    parser.add_argument('--models', default='./models', help='Path to model directory')
    parser.add_argument('--identity-lock', type=float, default=0.5, help='Identity preservation strength')
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return -1

    # Create engine
    print("Initializing UFRa engine...")
    engine = pyufra.create_engine()
    
    # Configure engine
    config = pyufra.ModelConfig()
    config.model_path = args.models
    config.backend = pyufra.GPUBackend.CUDA  # Try CUDA first
    config.batch_size = 1
    config.use_half_precision = True
    config.max_resolution = 1024

    if not engine.initialize(config):
        print("Error: Failed to initialize engine")
        return -1

    if not engine.load_models(args.models):
        print(f"Error: Failed to load models from {args.models}")
        return -1

    print(f"Engine initialized: {engine.get_version_info()}")

    # Set processing mode
    mode_map = {
        'feedforward': pyufra.ProcessingMode.FEEDFORWARD,
        'diffusion': pyufra.ProcessingMode.DIFFUSION,
        'hybrid': pyufra.ProcessingMode.HYBRID,
        'auto': pyufra.ProcessingMode.AUTO
    }
    engine.set_processing_mode(mode_map[args.mode])

    # Setup age controls
    controls = pyufra.AgeControls()
    controls.target_age = args.age
    controls.identity_lock_strength = args.identity_lock
    controls.temporal_stability = 0.8
    controls.texture_keep = 0.6
    controls.skin_clean = 0.4
    controls.enable_hair_aging = True
    controls.gray_density = 0.5

    # Process input
    if args.input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Process single image
        print(f"Processing image: {args.input}")
        process_image(engine, args.input, args.output, controls, args.preview)
    else:
        # Process video
        print(f"Processing video: {args.input}")
        process_video(engine, args.input, args.output, controls, args.preview)

    print("Processing complete!")
    return 0

def process_image(engine, input_path, output_path, controls, show_preview):
    """Process a single image"""
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not load image: {input_path}")
        return

    # Convert BGR to RGB for processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Setup frame context
    context = pyufra.FrameContext()
    context.frame_number = 0
    context.set_input_frame(rgb_image)
    context.controls = controls
    context.mode = engine.get_processing_mode()

    # Process frame
    result = engine.process_frame(context)
    
    if result.success:
        # Get output image
        output_rgb = result.get_output_frame()
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        
        # Save result
        cv2.imwrite(output_path, output_bgr)
        print(f"Output saved to: {output_path}")
        
        # Show preview if requested
        if show_preview:
            show_comparison(image, output_bgr, f"Age: {controls.target_age}")
            
        # Print metrics
        if result.metrics:
            print("Processing metrics:")
            for key, value in result.metrics.items():
                print(f"  {key}: {value}")
    else:
        print(f"Error processing image: {result.error_message}")

def process_video(engine, input_path, output_path, controls, show_preview):
    """Process a video file"""
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Setup frame context
            context = pyufra.FrameContext()
            context.frame_number = frame_number
            context.set_input_frame(rgb_frame)
            context.controls = controls
            context.mode = engine.get_processing_mode()

            # Process frame
            result = engine.process_frame(context)
            
            if result.success:
                # Convert back to BGR and write
                output_rgb = result.get_output_frame()
                output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
                out.write(output_bgr)
                
                # Show preview if requested
                if show_preview:
                    preview = cv2.resize(output_bgr, (640, 480))
                    cv2.imshow('UFRa Processing', preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                print(f"Warning: Failed to process frame {frame_number}: {result.error_message}")
                out.write(frame)  # Write original frame on failure

            # Progress update
            if frame_number % 30 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames})")

            frame_number += 1

    finally:
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"Video processing complete. Output saved to: {output_path}")

def show_comparison(original, processed, title):
    """Show side-by-side comparison of original and processed images"""
    # Resize images to fit screen
    max_height = 600
    if original.shape[0] > max_height:
        scale = max_height / original.shape[0]
        new_width = int(original.shape[1] * scale)
        original = cv2.resize(original, (new_width, max_height))
        processed = cv2.resize(processed, (new_width, max_height))

    # Create side-by-side comparison
    comparison = np.hstack([original, processed])
    
    # Add labels
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, f"Processed ({title})", (original.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('UFRa Comparison', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())