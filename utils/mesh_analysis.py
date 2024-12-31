import trimesh
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import sys

def check_manifold(mesh):
    edge_faces = mesh.face_adjacency_edges
    face_pairs = mesh.face_adjacency
    
    edge_face_count = {}
    for edge, faces in zip(edge_faces, face_pairs):
        edge_key = tuple(sorted(edge))
        edge_face_count[edge_key] = edge_face_count.get(edge_key, 0) + 1
    
    non_manifold_edge_count = sum(1 for count in edge_face_count.values() if count > 2)
    has_non_manifold_edges = non_manifold_edge_count > 0
    
    return has_non_manifold_edges, non_manifold_edge_count

def load_progress():
    if os.path.exists('analysis_progress.json'):
        with open('analysis_progress.json', 'r') as f:
            return json.load(f)
    return {
        'processed_files': [],
        'running_stats': {
            'total_files': 0,
            'vertex_sum': 0,
            'vertex_sq_sum': 0,
            'face_sum': 0,
            'face_sq_sum': 0,
            'volume_sum': 0,
            'empty_files': 0,
            'non_watertight': 0,
            'inconsistent_winding': 0,
            'non_manifold_files': 0,
            'files_with_non_manifold_edges': 0,
            'total_non_manifold_edges': 0
        }
    }

def save_progress(progress):
    with open('analysis_progress.json', 'w') as f:
        json.dump(progress, f)

def analyze_mesh_dataset(base_path, batch_size=20):
    # Load previous progress
    progress = load_progress()
    running_stats = progress['running_stats']
    processed_files = set(progress['processed_files'])
    
    # Get all files to process
    all_files = []
    for patient in os.listdir(base_path):
        patient_path = os.path.join(base_path, patient)
        for obj_file in os.listdir(patient_path):
            if obj_file.endswith('.obj'):
                file_path = os.path.join(patient_path, obj_file)
                all_files.append((patient, file_path))
    
    # Filter out already processed files
    remaining_files = [f for f in all_files if f[1] not in processed_files]
    
    print(f"Total files: {len(all_files)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"Remaining: {len(remaining_files)}")
    
    # Process in batches
    for i in range(0, len(remaining_files), batch_size):
        batch = remaining_files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} files)")
        
        for patient, file_path in tqdm(batch):
            try:
                mesh = trimesh.load_mesh(file_path)
                
                has_non_manifold_edges, non_manifold_edge_count = check_manifold(mesh)
                is_manifold = (mesh.is_watertight and 
                             mesh.is_winding_consistent and 
                             not has_non_manifold_edges)
                
                # Update running statistics
                running_stats['total_files'] += 1
                running_stats['vertex_sum'] += len(mesh.vertices)
                running_stats['vertex_sq_sum'] += len(mesh.vertices) ** 2
                running_stats['face_sum'] += len(mesh.faces)
                running_stats['face_sq_sum'] += len(mesh.faces) ** 2
                running_stats['volume_sum'] += mesh.volume if mesh.volume else 0
                running_stats['empty_files'] += 1 if mesh.is_empty else 0
                running_stats['non_watertight'] += 0 if mesh.is_watertight else 1
                running_stats['inconsistent_winding'] += 0 if mesh.is_winding_consistent else 1
                running_stats['non_manifold_files'] += 0 if is_manifold else 1
                running_stats['files_with_non_manifold_edges'] += 1 if has_non_manifold_edges else 0
                running_stats['total_non_manifold_edges'] += non_manifold_edge_count
                
                # Mark file as processed
                processed_files.add(file_path)
                
                # Free memory
                del mesh
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Save progress after each batch
        progress = {
            'processed_files': list(processed_files),
            'running_stats': running_stats
        }
        save_progress(progress)
        
        # Calculate and display current summary
        n = running_stats['total_files']
        current_summary = {
            'total_files': n,
            'average_vertices': running_stats['vertex_sum'] / n if n > 0 else 0,
            'std_vertices': np.sqrt((running_stats['vertex_sq_sum'] / n) - (running_stats['vertex_sum'] / n) ** 2) if n > 0 else 0,
            'average_faces': running_stats['face_sum'] / n if n > 0 else 0,
            'std_faces': np.sqrt((running_stats['face_sq_sum'] / n) - (running_stats['face_sum'] / n) ** 2) if n > 0 else 0,
            'average_volume': running_stats['volume_sum'] / n if n > 0 else 0,
            'empty_files': running_stats['empty_files'],
            'non_watertight': running_stats['non_watertight'],
            'inconsistent_winding': running_stats['inconsistent_winding'],
            'non_manifold_files': running_stats['non_manifold_files'],
            'files_with_non_manifold_edges': running_stats['files_with_non_manifold_edges'],
            'total_non_manifold_edges': running_stats['total_non_manifold_edges']
        }
        
        print("\nCurrent Summary:")
        for key, value in current_summary.items():
            print(f"{key}: {value}")

        break
    
    pd.DataFrame([current_summary]).to_csv('mesh_analysis_summary_part1.csv', index=False)
    return current_summary
if __name__ == "__main__":
    path = sys.argv[1]
    summary = analyze_mesh_dataset(path, batch_size=60)