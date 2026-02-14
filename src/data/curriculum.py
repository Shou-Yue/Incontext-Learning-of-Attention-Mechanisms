"""
Curriculum learning for gradually increasing task complexity during training.
"""


class Curriculum:
    """
    Manages curriculum learning by gradually increasing:
    1. Number of dimensions (n_dims_truncated)
    2. Number of in-context examples (n_points)
    """
    
    def __init__(self, 
                 dims_start=5, dims_end=20, dims_inc=1, dims_interval=2000,
                 points_start=11, points_end=41, points_inc=2, points_interval=2000):
        """
        Args:
            dims_start: Starting number of dimensions
            dims_end: Final number of dimensions
            dims_inc: Increment in dimensions
            dims_interval: Steps between increments
            points_start: Starting number of points (in-context examples + 1 query)
            points_end: Final number of points
            points_inc: Increment in points
            points_interval: Steps between increments
        """
        self.n_dims_truncated = dims_start
        self.n_points = points_start
        
        self.dims_start = dims_start
        self.dims_end = dims_end
        self.dims_inc = dims_inc
        self.dims_interval = dims_interval
        
        self.points_start = points_start
        self.points_end = points_end
        self.points_inc = points_inc
        self.points_interval = points_interval
        
        self.step_count = 0
    
    def update(self):
        """Update curriculum after each training step."""
        self.step_count += 1
        
        # Update dimensions
        if self.step_count % self.dims_interval == 0:
            self.n_dims_truncated = min(
                self.n_dims_truncated + self.dims_inc,
                self.dims_end
            )
        
        # Update points
        if self.step_count % self.points_interval == 0:
            self.n_points = min(
                self.n_points + self.points_inc,
                self.points_end
            )
    
    def get_state(self):
        """Get current curriculum state for logging."""
        return {
            'n_dims_truncated': self.n_dims_truncated,
            'n_points': self.n_points,
            'step_count': self.step_count
        }
