class Curriculum:
    def __init__(self, dims_start, dims_end, dims_inc, points_start, points_end, points_inc, interval):
        # store current values, which will be used during training
        self.dims = dims_start
        self.points = points_start

        self.dims_start = dims_start
        self.dims_end = dims_end
        self.dims_inc = dims_inc

        self.points_start = points_start
        self.points_end = points_end
        self.points_inc = points_inc

        self.interval = interval
        self.step = 0

    def update(self):
        """
        Updates curriculum at the specified interval
        """
        self.step += 1

        if self.step % self.interval == 0:
            self.dims = min(self.dims + self.dims_inc, self.dims_end)
            self.points = min(self.points + self.points_inc, self.points_end)