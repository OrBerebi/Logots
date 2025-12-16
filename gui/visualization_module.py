# visualization_module.py

import pandas as pd
import numpy as np
import os
import cv2 
from PIL import Image, ImageDraw, ImageFont
import ast 

def visualize_mrt_delta_position(
    mrt_df, 
    vis_df, 
    video_path, 
    output_gif_path, 
    img_size=640
):
    """
    Generates a GIF by overlaying data on the raw MP4 video file.
    Includes strict type casting to prevent NumPy 'ambiguous truth value' errors.
    """
    
    # --- File Check ---
    if not os.path.exists(video_path):
        print(f" CRITICAL ERROR: Video file not found at path: {video_path}")
        return False
    
    # --- Prepare Lookup Tables ---
    try:
        if mrt_df.empty or 'experience_id' not in mrt_df.columns:
             return False
        mrt_lookup = mrt_df.set_index('experience_id')
    except Exception as e:
         print(f"Error indexing Mart data: {e}")
         return False

    try:
        # Fallback if frame_id is missing
        if 'frame_id' not in vis_df.columns:
            vis_df = vis_df.copy()
            vis_df['frame_id'] = vis_df.index 
        vis_lookup = vis_df.set_index('frame_id')
    except Exception as e:
         print(f"Error indexing Visual data: {e}")
         return False

    output_frames = []
    print(f"Processing video from {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return False

    font = ImageFont.load_default()
    frame_id = 0 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 
        
        # Resize and Convert
        frame = cv2.resize(frame, (img_size, img_size))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)

        delta_text = "delta_cat_position: N/A (no data)"
        start_point = None
        end_point = None

        # --- VISUAL LAYER: Draw Bounding Box ---
        if frame_id in vis_lookup.index:
            vis_row = vis_lookup.loc[frame_id]
            if isinstance(vis_row, pd.DataFrame): vis_row = vis_row.iloc[0] 

            # 1. Safe Extraction
            primary_centroid = vis_row.get('bounding_box_centroid', None)
            detections = vis_row.get('raw_detection', [])

            # 2. Sanitization
            if isinstance(detections, str):
                 try: detections = ast.literal_eval(detections)
                 except: detections = []
            elif isinstance(detections, np.ndarray):
                 detections = detections.tolist()
            if not isinstance(detections, list):
                 detections = []

            if isinstance(primary_centroid, str):
                try: primary_centroid = ast.literal_eval(primary_centroid)
                except: primary_centroid = None
            
            # 3. Safe Execution
            # 'primary_centroid is not None' prevents the array error
            if primary_centroid is not None and len(detections) > 0:
                for det in detections:
                    det_centroid = det.get('bounding_box_centroid')
                    
                    is_match = False
                    try:
                        # Force comparison using numpy's safe tool
                        c1 = np.array(det_centroid) if det_centroid is not None else np.array([])
                        c2 = np.array(primary_centroid)
                        if np.array_equal(c1, c2):
                            is_match = True
                    except:
                        pass 

                    if is_match:
                        # Force floats for drawing coordinates
                        xmin = float(det['xmin'])
                        ymin = float(det['ymin'])
                        xmax = float(det['xmax'])
                        ymax = float(det['ymax'])
                        draw.rectangle([xmin, ymin, xmax, ymax], outline="cyan", width=2)
                        break 
        
        # --- MART LAYER: Draw Vectors ---
        if frame_id in mrt_lookup.index:
            mrt_row = mrt_lookup.loc[frame_id]
            if isinstance(mrt_row, pd.DataFrame): mrt_row = mrt_row.iloc[0]

            delta_val = mrt_row['delta_cat_position']
            
            if isinstance(delta_val, str):
                try: delta_val = ast.literal_eval(delta_val)
                except: delta_val = None

            if pd.isna(mrt_row['timestamp']):
                delta_text = "delta_cat_position: N/A (window filling)"
            
            # Check length safely
            elif delta_val is not None and hasattr(delta_val, '__len__') and len(delta_val) == 2:
                try:
                    # --- CRITICAL FIX: Force everything to standard Python floats ---
                    # This strips away any NumPy array wrappers that cause crashes
                    dx = float(delta_val[0])
                    dy = float(delta_val[1])
                    end_x = float(mrt_row['cat_position_x'])
                    end_y = float(mrt_row['cat_position_y'])

                    if any(pd.isna([dx, dy, end_x, end_y])):
                         delta_text = "delta_cat_position: N/A (cat not detected)"
                    else:
                        delta_text = f"delta_cat_position: ({dx:.1f}, {dy:.1f})"
                        # Now creating tuples from pure floats is safe
                        end_point = (end_x, end_y)
                        start_point = (end_x - dx, end_y - dy)
                except (ValueError, TypeError):
                    delta_text = "delta_cat_position: Error"
            else:
                delta_text = f"delta_cat_position: {str(delta_val)}"
        
        # --- DRAWING VECTORS ---
        # Safe check: "is not None" is mandatory when variables might have been arrays in previous iterations
        if start_point is not None and end_point is not None:
            try:
                sp = (int(start_point[0]), int(start_point[1]))
                ep = (int(end_point[0]), int(end_point[1]))
                
                draw.ellipse((sp[0]-5, sp[1]-5, sp[0]+5, sp[1]+5), fill="lime", outline="lime")
                draw.line([sp, ep], fill="red", width=3)
                draw.ellipse((ep[0]-5, ep[1]-5, ep[0]+5, ep[1]+5), fill="red", outline="red")
            except Exception:
                pass 

        draw.text((10, 10), f"Frame: {frame_id}", fill="white", font=font)
        draw.text((10, 30), delta_text, fill="yellow", font=font)
        
        output_frames.append(pil_img)
        frame_id += 1 

    cap.release()
    print("...Processing complete.")
    
    if not output_frames:
        print("No frames were processed.")
        return False

    print(f"Encoding animated GIF to {output_gif_path}...")
    try:
        os.makedirs(os.path.dirname(output_gif_path) or '.', exist_ok=True)
        output_frames[0].save(output_gif_path, format='GIF',
                              save_all=True, append_images=output_frames[1:], 
                              duration=150, 
                              loop=0)
        print(f"✅ Success! Annotated video saved as: {output_gif_path}")
        return True
    except Exception as e:
        print(f"🛑 Error saving GIF file: {e}")
        return False

def create_annotated_video(mrt_df, trans_visual_df, video_file_path, output_gif_path):
    return visualize_mrt_delta_position(
        mrt_df=mrt_df, 
        vis_df=trans_visual_df, 
        video_path=video_file_path, 
        output_gif_path=output_gif_path
    )