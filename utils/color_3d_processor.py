"""
3D Color Data Processor for ColorAnalyzer
Processes image color data for 3D visualization
"""

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import json
import random
from typing import List, Dict, Tuple, Any


class Color3DProcessor:
    """Processes color data for 3D visualization"""
    
    def __init__(self, max_points: int = 10000, n_clusters: int = 8):
        self.max_points = max_points
        self.n_clusters = n_clusters
    
    def process_image_for_3d(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process an image and extract color data for 3D visualization
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing 3D visualization data
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Ensure RGB format
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # Convert RGBA to RGB
            img_array = img_array[:, :, :3]
        elif len(img_array.shape) == 2:
            # Convert grayscale to RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Extract color data
        color_data = self._extract_color_points(img_array)
        
        # Perform clustering
        cluster_data = self._perform_clustering(img_array)
        
        # Calculate statistics
        stats = self._calculate_color_statistics(img_array, cluster_data)
        
        # Generate HSV data
        hsv_data = self._convert_to_hsv_coordinates(color_data)
        
        return {
            'rgb_points': color_data,
            'hsv_points': hsv_data,
            'clusters': cluster_data,
            'statistics': stats,
            'metadata': {
                'total_pixels': img_array.shape[0] * img_array.shape[1],
                'sampled_points': len(color_data),
                'n_clusters': len(cluster_data)
            }
        }
    
    def _extract_color_points(self, img_array: np.ndarray) -> List[Dict[str, int]]:
        """Extract color points from image, sampling if necessary"""
        # Reshape to get all pixels
        pixels = img_array.reshape(-1, 3)
        
        # Sample points if too many
        if len(pixels) > self.max_points:
            indices = np.random.choice(len(pixels), self.max_points, replace=False)
            pixels = pixels[indices]
        
        # Convert to list of dictionaries
        color_points = []
        for pixel in pixels:
            color_points.append({
                'r': int(pixel[0]),
                'g': int(pixel[1]),
                'b': int(pixel[2])
            })
        
        return color_points
    
    def _perform_clustering(self, img_array: np.ndarray) -> List[Dict[str, Any]]:
        """Perform K-means clustering on image colors"""
        # Reshape image for clustering
        pixels = img_array.reshape(-1, 3)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers and labels
        centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Calculate percentages
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)
        
        # Create cluster data
        clusters = []
        for i, (center, percentage) in enumerate(zip(centers, percentages)):
            # Calculate cluster size based on percentage (for visualization)
            size = max(5, min(20, int(percentage * 100)))
            
            clusters.append({
                'r': int(center[0]),
                'g': int(center[1]),
                'b': int(center[2]),
                'percentage': float(percentage),
                'size': size,
                'hex': f"#{center[0]:02x}{center[1]:02x}{center[2]:02x}",
                'cluster_id': i
            })
        
        # Sort by percentage (largest first)
        clusters.sort(key=lambda x: x['percentage'], reverse=True)
        
        return clusters
    
    def _convert_to_hsv_coordinates(self, rgb_points: List[Dict[str, int]]) -> List[Dict[str, float]]:
        """Convert RGB points to HSV cylindrical coordinates for 3D visualization"""
        hsv_points = []
        
        for point in rgb_points:
            # Convert RGB to HSV
            rgb_array = np.array([[[point['r'], point['g'], point['b']]]], dtype=np.uint8)
            hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
            h, s, v = hsv_array[0, 0]
            
            # Convert to cylindrical coordinates
            # H: 0-179 -> 0-2π radians
            # S: 0-255 -> 0-1 (radius)
            # V: 0-255 -> 0-1 (height)
            h_rad = (h / 179.0) * 2 * np.pi
            s_norm = s / 255.0
            v_norm = v / 255.0
            
            # Cylindrical to Cartesian conversion
            x = s_norm * np.cos(h_rad) * 100  # Scale for visualization
            y = v_norm * 100  # Height
            z = s_norm * np.sin(h_rad) * 100  # Scale for visualization
            
            hsv_points.append({
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'h': float(h),
                's': float(s),
                'v': float(v),
                'r': point['r'],
                'g': point['g'],
                'b': point['b']
            })
        
        return hsv_points
    
    def _calculate_color_statistics(self, img_array: np.ndarray, clusters: List[Dict]) -> Dict[str, Any]:
        """Calculate various color statistics"""
        pixels = img_array.reshape(-1, 3)
        
        # Basic statistics
        stats = {
            'mean_rgb': {
                'r': float(np.mean(pixels[:, 0])),
                'g': float(np.mean(pixels[:, 1])),
                'b': float(np.mean(pixels[:, 2]))
            },
            'std_rgb': {
                'r': float(np.std(pixels[:, 0])),
                'g': float(np.std(pixels[:, 1])),
                'b': float(np.std(pixels[:, 2]))
            },
            'dominant_color': clusters[0] if clusters else None,
            'color_diversity': self._calculate_color_diversity(pixels),
            'brightness_avg': float(np.mean(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))),
            'contrast': self._calculate_contrast(img_array)
        }
        
        return stats
    
    def _calculate_color_diversity(self, pixels: np.ndarray) -> float:
        """Calculate color diversity metric"""
        # Quantize colors to reduce noise
        quantized = (pixels // 8) * 8
        unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
        
        # Diversity as ratio of unique colors to total pixels
        diversity = len(unique_colors) / len(pixels)
        return float(min(1.0, diversity * 10))  # Scale for better visualization
    
    def _calculate_contrast(self, img_array: np.ndarray) -> float:
        """Calculate image contrast using standard deviation of luminance"""
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate contrast as standard deviation
        contrast = float(np.std(gray) / 255.0)
        return contrast


def create_3d_visualization_data(image: Image.Image, max_points: int = 10000) -> Dict[str, Any]:
    """
    Convenience function to create 3D visualization data from an image
    
    Args:
        image: PIL Image object
        max_points: Maximum number of points to sample for visualization
        
    Returns:
        Dictionary containing all 3D visualization data
    """
    processor = Color3DProcessor(max_points=max_points)
    return processor.process_image_for_3d(image)


def save_3d_data_to_json(data: Dict[str, Any], filepath: str) -> None:
    """Save 3D visualization data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_3d_data_from_json(filepath: str) -> Dict[str, Any]:
    """Load 3D visualization data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
