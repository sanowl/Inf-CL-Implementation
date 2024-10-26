def xorshift32(state):
    # XORShift PRNG implementation
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= (state >> 17) & 0xFFFFFFFF
    state ^= (state << 5) & 0xFFFFFFFF
    return state

def fast_rand():
    # Generate random float between 0 and 1
    global _random_state
    _random_state = xorshift32(_random_state)
    return (_random_state & 0xFFFFFF) / 0x1000000

def fast_sqrt(x):
    # Fast square root approximation
    if x <= 0:
        return 0
    
    estimate = x
    for _ in range(10):  # Newton's method
        estimate = (estimate + x / estimate) * 0.5
    return estimate

def fast_exp(x):
    # Fast exponential approximation
    if x > 88.0:
        return float('inf')
    if x < -88.0:
        return 0.0
    
    # Use Taylor series expansion
    result = 1.0
    term = 1.0
    for i in range(1, 12):  # 12 terms for good precision
        term *= x / i
        result += term
    return result

def fast_log(x):
    # Fast natural logarithm approximation
    if x <= 0:
        return float('-inf')
        
    # Normalize x to [1,2)
    exp = 0
    while x >= 2.0:
        x *= 0.5
        exp += 1
    while x < 1.0:
        x *= 2.0
        exp -= 1
        
    # Use Taylor series around 1
    x_minus_1 = x - 1.0
    result = x_minus_1
    term = x_minus_1
    for i in range(2, 10):
        term *= -x_minus_1 * (i-1) / i
        result += term
        
    return result + 0.693147180559945 * exp  # ln(2) * exp

class FastArray:
    def __init__(self, size):
        self.size = size
        self.data = [0.0] * size
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def __setitem__(self, idx, val):
        self.data[idx] = val

class FastMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = FastArray(rows * cols)
        
    def get(self, row, col):
        return self.data[row * self.cols + col]
        
    def set(self, row, col, val):
        self.data[row * self.cols + col] = val
        
    def init_random(self):
        # Xavier initialization
        scale = fast_sqrt(6.0 / (self.rows + self.cols))
        for i in range(self.rows * self.cols):
            self.data[i] = (fast_rand() * 2 - 1) * scale

class TileProcessor:
    def __init__(self, batch_size, tile_size):
        self.batch_size = batch_size
        self.tile_size = tile_size
        
    def process_tile(self, img_features, txt_features, start_row, start_col):
        tile_rows = min(self.tile_size, img_features.rows - start_row)
        tile_cols = min(self.tile_size, txt_features.rows - start_col)
        
        similarity = FastMatrix(tile_rows, tile_cols)
        
        # Compute similarities for this tile
        for i in range(tile_rows):
            for j in range(tile_cols):
                dot_product = 0.0
                for k in range(img_features.cols):
                    dot_product += (img_features.get(start_row + i, k) * 
                                  txt_features.get(start_col + j, k))
                similarity.set(i, j, dot_product)
                
        return similarity

    def compute_tile_lse(self, similarity):
        result = FastArray(similarity.rows)
        for i in range(similarity.rows):
            # Find max for numerical stability
            max_val = float('-inf')
            for j in range(similarity.cols):
                max_val = max(max_val, similarity.get(i, j))
                
            # Compute log-sum-exp
            sum_exp = 0.0
            for j in range(similarity.cols):
                sum_exp += fast_exp(similarity.get(i, j) - max_val)
            
            result[i] = max_val + fast_log(sum_exp)
            
        return result

class InfCL:
    def __init__(self, batch_size, num_gpus, tile_size, feature_dim):
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.tile_size = tile_size
        self.feature_dim = feature_dim
        self.processor = TileProcessor(batch_size, tile_size)
        
    def normalize_features(self, features):
        for i in range(features.rows):
            norm = 0.0
            for j in range(features.cols):
                val = features.get(i, j)
                norm += val * val
            norm = fast_sqrt(norm)
            if norm > 0:
                for j in range(features.cols):
                    features.set(i, j, features.get(i, j) / norm)
    
    def forward(self):
        # Initialize features
        img_features = FastMatrix(self.batch_size, self.feature_dim)
        txt_features = FastMatrix(self.batch_size, self.feature_dim)
        
        # Random initialization
        img_features.init_random()
        txt_features.init_random()
        
        # Normalize features
        self.normalize_features(img_features)
        self.normalize_features(txt_features)
        
        # Initialize accumulators
        global_lse = FastArray(self.batch_size)
        for i in range(self.batch_size):
            global_lse[i] = float('-inf')
        
        positive_sum = 0.0
        
        # Process tiles
        num_tiles = (self.batch_size + self.tile_size - 1) // self.tile_size
        
        for gpu_id in range(self.num_gpus):
            start_idx = gpu_id * (self.batch_size // self.num_gpus)
            end_idx = start_idx + (self.batch_size // self.num_gpus)
            
            for i in range(num_tiles):
                row_start = i * self.tile_size
                if row_start >= end_idx or row_start < start_idx:
                    continue
                    
                row_lse = FastArray(self.tile_size)
                for j in range(len(row_lse.data)):
                    row_lse[j] = float('-inf')
                
                for j in range(num_tiles):
                    # Process single tile
                    similarity = self.processor.process_tile(
                        img_features, txt_features, row_start, j * self.tile_size)
                    
                    # Compute LSE for this tile
                    local_lse = self.processor.compute_tile_lse(similarity)
                    
                    # Merge LSE results
                    for k in range(similarity.rows):
                        if row_lse[k] == float('-inf'):
                            row_lse[k] = local_lse[k]
                        else:
                            max_val = max(row_lse[k], local_lse[k])
                            row_lse[k] = max_val + fast_log(
                                fast_exp(row_lse[k] - max_val) + 
                                fast_exp(local_lse[k] - max_val))
                    
                    # Collect positive pairs if on diagonal
                    if i == j:
                        for k in range(min(self.tile_size, 
                                         similarity.rows, 
                                         similarity.cols)):
                            positive_sum += similarity.get(k, k)
                
                # Update global LSE
                for k in range(self.tile_size):
                    if row_start + k < end_idx and row_start + k >= start_idx:
                        global_lse[row_start + k] = row_lse[k]
        
        # Compute final loss
        loss = -positive_sum / self.batch_size
        lse_sum = sum(x for x in global_lse.data if x != float('-inf'))
        loss += lse_sum / self.batch_size
        
        return loss

# Initialize random state
_random_state = 12345

def run_demo():
    # Set parameters
    batch_size = 32
    feature_dim = 8
    num_gpus = 2
    tile_size = 8
    
    # Create and run InfCL
    infcl = InfCL(batch_size, num_gpus, tile_size, feature_dim)
    loss = infcl.forward()
    
    print(f"Contrastive Loss: {loss}")

if __name__ == "__main__":
    run_demo()