"""
GFA Compression API - Python wrapper for the C++ compression library.
"""
import os
import sys
import time

# Add the build directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'build'))
import gfa_compression as gfa_lib


class GFACompressor:
    """Stateful Python wrapper for the GFA compression library."""

    def __init__(self, gfa_file_path: str):
        """Initialize compressor with path to a GFA file."""
        if not os.path.exists(gfa_file_path):
            raise FileNotFoundError(f"GFA file not found: {gfa_file_path}")

        self.gfa_file_path = gfa_file_path
        self.compressed_data = None
        self.compressed_data_gpu = None
        self.is_compressed = False
        self.compression_time = None
        self.decompression_time = None
        self.input_file_size = os.path.getsize(gfa_file_path)

    def _resolve_params(self, freq_threshold, delta_round):
        """Resolve compression parameters from args or environment variables."""
        threshold = freq_threshold
        if threshold is None:
            env_val = os.environ.get('GFA_COMPRESSION_FREQ_THRESHOLD', '')
            threshold = int(env_val) if env_val.isdigit() else 2
        
        delta = delta_round
        if delta is None:
            env_val = os.environ.get('GFA_COMPRESSION_DELTA_ROUNDS', '')
            delta = int(env_val) if env_val.isdigit() else 1
        
        return threshold, delta

    def compress(self, num_rounds: int = 8, freq_threshold: int = None, delta_round: int = None, num_threads: int = None):
        """
        Compress GFA file using 2-mer grammar compression.

        Args:
            num_rounds: Number of compression rounds (default: 8)
            freq_threshold: Minimum 2-mer frequency to create rule (default: 2)
            delta_round: Number of delta encoding rounds (default: 1)
            num_threads: Number of threads to use (default: None, use all available)
        """
        threshold, delta = self._resolve_params(freq_threshold, delta_round)
        threads = num_threads if num_threads is not None else 0

        start_time = time.perf_counter()
        # compress_gfa now returns CompressedData directly
        self.compressed_data = gfa_lib.compress(
            self.gfa_file_path,
            num_rounds=num_rounds,
            freq_threshold=threshold,
            delta_round=delta,
            num_threads=threads
        )
        self.compression_time = time.perf_counter() - start_time
        self.is_compressed = True

        throughput = (self.input_file_size / (1024 * 1024)) / self.compression_time if self.compression_time > 0 else 0
        print(f"Time: {self.compression_time:.2f}s ({throughput:.2f} MB/s)")

    def decompress(self, num_threads: int = None):
        """Decompress and return the reconstructed GfaGraph.
        
        Args:
             num_threads: Number of threads to use (default: None, use all available)
        """
        if not self.is_compressed:
            print("Error: No compressed data. Call compress() first.")
            return None
        
        threads = num_threads if num_threads is not None else 0
        
        start_time = time.perf_counter()
        decompressed_graph = gfa_lib.decompress(self.compressed_data, num_threads=threads)
        self.decompression_time = time.perf_counter() - start_time
        
        throughput = (self.input_file_size / (1024 * 1024)) / self.decompression_time if self.decompression_time > 0 else 0
        print(f"Time: {self.decompression_time:.2f}s ({throughput:.2f} MB/s)")
        return decompressed_graph

    def save(self, output_path: str):
        """Save compressed data to binary file (.gfaz)."""
        if not self.is_compressed:
            print("Error: No compressed data. Call compress() first.")
            return False
        gfa_lib.serialize(self.compressed_data, output_path)
        return True

    @classmethod
    def load(cls, input_path: str):
        """Load compressed data from binary file (.gfaz)."""
        compressed_data = gfa_lib.deserialize(input_path)

        instance = cls.__new__(cls)
        instance.gfa_file_path = None
        instance.compressed_data = compressed_data
        instance.compressed_data_gpu = None
        instance.is_compressed = True
        instance.compression_time = 0
        instance.decompression_time = 0
        instance.input_file_size = 0
        return instance

    def write_gfa(self, graph, output_path: str = None):
        """Write GfaGraph to GFA file."""
        if output_path is None:
            if self.gfa_file_path:
                output_path = self.gfa_file_path + ".decompressed"
            else:
                raise ValueError("No output path specified")
        gfa_lib.write_gfa(graph, output_path)
        return output_path

    def verify(self, decompressed_graph):
        """Verify decompressed graph by re-parsing the original file."""
        if not self.gfa_file_path:
            print("Error: No original GFA file path (loaded from .gfaz).")
            return False
        original_graph = gfa_lib.parse(self.gfa_file_path)
        return gfa_lib.verify_round_trip(original_graph, decompressed_graph)

    def print_summary(self):
        """Print compression summary statistics."""
        if not self.is_compressed:
            print("Error: Not compressed yet. Call compress() first.")
            return

        print("\n--- Compressed Data Stats ---")

        # Rules info
        if self.compressed_data and self.compressed_data.layer_rule_ranges:
            total_rules = sum(r.end_id - r.start_id for r in self.compressed_data.layer_rule_ranges)
            print(f"\n--- Master Rulebook ---")
            print(f"  Total rules: {total_rules}")
            for layer in self.compressed_data.layer_rule_ranges:
                print(f"    K={layer.k}: [{layer.start_id}, {layer.end_id})")

            first_sz = len(self.compressed_data.rules_first_zstd.payload) if self.compressed_data.rules_first_zstd.payload else 0
            second_sz = len(self.compressed_data.rules_second_zstd.payload) if self.compressed_data.rules_second_zstd.payload else 0
            print(f"  ZSTD rules: first={first_sz} Bytes, second={second_sz} Bytes")

        # Paths info
        num_paths = len(self.compressed_data.sequence_lengths) if self.compressed_data.sequence_lengths else 0
        paths_sz = len(self.compressed_data.paths_zstd.payload) if self.compressed_data.paths_zstd.payload else 0
        print(f"\n--- Paths ---")
        print(f"  Count: {num_paths}")
        print(f"  ZSTD payload: {paths_sz} Bytes")

        # Walks info
        num_walks = len(self.compressed_data.walk_lengths) if self.compressed_data.walk_lengths else 0
        walks_sz = len(self.compressed_data.walks_zstd.payload) if self.compressed_data.walks_zstd.payload else 0
        print(f"\n--- Walks ---")
        print(f"  Count: {num_walks}")
        print(f"  ZSTD payload: {walks_sz} Bytes")

    # --- GPU Methods (require CUDA build) ---
    
    def compress_gpu(self, num_rounds: int = 8):
        """GPU-accelerated compression using nvComp."""
        start_time = time.perf_counter()
        self.compressed_data_gpu = gfa_lib.compress_gfa_gpu(self.gfa_file_path, num_rounds=num_rounds)
        self.compression_time = time.perf_counter() - start_time
        self.is_compressed = True
        
        throughput = (self.input_file_size / (1024 * 1024)) / self.compression_time if self.compression_time > 0 else 0
        print(f"[GPU] Time: {self.compression_time:.2f}s ({throughput:.2f} MB/s)")
        print(f"[GPU] Rules: {self.compressed_data_gpu.total_rules():,}")
        return self.compressed_data_gpu

    def decompress_gpu(self):
        """GPU-accelerated decompression."""
        if self.compressed_data_gpu is None:
            print("Error: No GPU data. Call compress_gpu() first.")
            return None
        
        start_time = time.perf_counter()
        result = gfa_lib.decompress_to_gpu_layout(self.compressed_data_gpu)
        self.decompression_time = time.perf_counter() - start_time
        
        print(f"[GPU] Time: {self.decompression_time:.2f}s")
        print(f"[GPU] Segments: {result.num_segments:,}, Paths: {result.num_paths:,}, Links: {result.num_links:,}")
        return result

    def verify_gpu(self, original, decompressed):
        """Verify GPU round-trip."""
        return gfa_lib.verify_gpu_round_trip(original, decompressed)

    def compress_decompress_verify_gpu(self, num_rounds: int = 8):
        """Full GPU round-trip test."""
        print("=" * 50)
        print("GPU Round-Trip Test")
        print("=" * 50)
        
        graph = gfa_lib.parse(self.gfa_file_path)
        original = gfa_lib.convert_to_gpu_layout(graph)
        print(f"Original: {original.num_segments:,} seg, {original.num_paths:,} paths, {original.num_links:,} links")
        
        self.compress_gpu(num_rounds=num_rounds)
        decompressed = self.decompress_gpu()
        success = self.verify_gpu(original, decompressed)
        
        print("=" * 50)
        print("✓ PASSED" if success else "✗ FAILED")
        print("=" * 50)
        return success
