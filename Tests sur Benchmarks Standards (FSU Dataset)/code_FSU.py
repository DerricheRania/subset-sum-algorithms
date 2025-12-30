import time
import tracemalloc
from itertools import combinations
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import os
import re
import csv
from datetime import datetime

class SubsetSum:
    
    def __init__(self, arr: List[int], target: int, instance_name: str = "Unknown"):
        self.arr = arr
        self.target = target
        self.n = len(arr)
        self.instance_name = instance_name
    
    # ========== BRUTE FORCE APPROACH ==========
    def brute_force(self) -> Tuple[bool, Optional[List[int]]]:
        """Brute force solution: Try all possible subsets."""
        for size in range(1, self.n + 1):
            for subset in combinations(range(self.n), size):
                current_sum = sum(self.arr[i] for i in subset)
                if current_sum == self.target:
                    solution = [self.arr[i] for i in subset]
                    return True, solution
        return False, None
    
    # ========== DYNAMIC PROGRAMMING APPROACH ==========
    def dynamic_programming(self) -> Tuple[bool, Optional[List[int]]]:
        """Dynamic Programming solution using tabulation."""
        dp = [[False for _ in range(self.target + 1)] for _ in range(self.n + 1)]
        
        for i in range(self.n + 1):
            dp[i][0] = True
        
        for i in range(1, self.n + 1):
            for j in range(1, self.target + 1):
                dp[i][j] = dp[i-1][j]
                if j >= self.arr[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j - self.arr[i-1]]
        
        if not dp[self.n][self.target]:
            return False, None
        
        subset = []
        i, j = self.n, self.target
        while i > 0 and j > 0:
            if dp[i-1][j]:
                i -= 1
            else:
                subset.append(self.arr[i-1])
                j -= self.arr[i-1]
                i -= 1
        
        return True, subset
    
    # ========== OPTIMIZED DP (Space-Optimized) ==========
    def dp_space_optimized(self) -> Tuple[bool, Optional[List[int]]]:
        """Space-optimized DP solution using 1D array."""
        dp = [False] * (self.target + 1)
        dp[0] = True
        
        for num in self.arr:
            for j in range(self.target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        
        return dp[self.target], None
    
    # ========== VERIFICATION FUNCTION ==========
    @staticmethod
    def verify_solution(arr: List[int], target: int, subset: Optional[List[int]]) -> bool:
        """Verify if the subset is a valid solution."""
        if subset is None:
            return False
        
        arr_copy = arr.copy()
        for element in subset:
            if element not in arr_copy:
                return False
            arr_copy.remove(element)
        
        return sum(subset) == target
    
    # ========== PERFORMANCE COMPARISON ==========
    def compare_algorithms(self) -> dict:
        """Compare brute force and DP approaches in terms of time and memory."""
        results = {}
        
        # ===== BRUTE FORCE =====
        print("Running Brute Force...")
        tracemalloc.start()
        start_time = time.perf_counter()
        
        bf_exists, bf_subset = self.brute_force()
        
        bf_time = time.perf_counter() - start_time
        bf_current, bf_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['brute_force'] = {
            'solution_exists': bf_exists,
            'subset': bf_subset,
            'time_seconds': bf_time,
            'memory_current_kb': bf_current / 1024,
            'memory_peak_kb': bf_peak / 1024,
            'verified': self.verify_solution(self.arr, self.target, bf_subset)
        }
        
        # ===== DYNAMIC PROGRAMMING =====
        print("Running Dynamic Programming...")
        tracemalloc.start()
        start_time = time.perf_counter()
        
        dp_exists, dp_subset = self.dynamic_programming()
        
        dp_time = time.perf_counter() - start_time
        dp_current, dp_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['dynamic_programming'] = {
            'solution_exists': dp_exists,
            'subset': dp_subset,
            'time_seconds': dp_time,
            'memory_current_kb': dp_current / 1024,
            'memory_peak_kb': dp_peak / 1024,
            'verified': self.verify_solution(self.arr, self.target, dp_subset)
        }
        
        # ===== SPACE-OPTIMIZED DP =====
        print("Running Space-Optimized DP...")
        tracemalloc.start()
        start_time = time.perf_counter()
        
        dp_opt_exists, _ = self.dp_space_optimized()
        
        dp_opt_time = time.perf_counter() - start_time
        dp_opt_current, dp_opt_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['dp_space_optimized'] = {
            'solution_exists': dp_opt_exists,
            'subset': None,
            'time_seconds': dp_opt_time,
            'memory_current_kb': dp_opt_current / 1024,
            'memory_peak_kb': dp_opt_peak / 1024,
            'verified': 'N/A (only checks existence)'
        }
        
        results['speedup'] = {
            'dp_vs_brute_force': bf_time / dp_time if dp_time > 0 else float('inf'),
            'space_optimized_vs_brute_force': bf_time / dp_opt_time if dp_opt_time > 0 else float('inf')
        }
        
        return results
    
    @staticmethod
    def print_results(results: dict):
        print("\n" + "="*70)
        print("SUBSET SUM - PERFORMANCE COMPARISON")
        print("="*70)
        
        for algo_name, metrics in results.items():
            if algo_name == 'speedup':
                continue
                
            print(f"\n{algo_name.upper().replace('_', ' ')}:")
            print("-" * 70)
            print(f"  Solution Exists: {metrics['solution_exists']}")
            if metrics['subset'] is not None:
              print(f"  Subset weights: {metrics['subset']}")
              print(f"  Subset sum: {sum(metrics['subset'])}")
            else:
              print("  Subset: None")
            print(f"  Time: {metrics['time_seconds']:.6f} seconds")
            print(f"  Memory (Peak): {metrics['memory_peak_kb']:.2f} KB")
            print(f"  Verified: {metrics['verified']}")
        
        print("\n" + "="*70)
        print("SPEEDUP ANALYSIS:")
        print("-" * 70)
        print(f"  DP vs Brute Force: {results['speedup']['dp_vs_brute_force']:.2f}x faster")
        print(f"  Space-Optimized DP vs Brute Force: {results['speedup']['space_optimized_vs_brute_force']:.2f}x faster")
        print("="*70 + "\n")
    
    @staticmethod
    def plot_comparison(results: dict, instance_name: str = ""):
        algorithms = ['Brute Force', 'Dynamic\nProgramming', 'Space-Optimized\nDP']
        
        times = [
            results['brute_force']['time_seconds'] * 1000,
            results['dynamic_programming']['time_seconds'] * 1000,
            results['dp_space_optimized']['time_seconds'] * 1000
        ]
        
        memory = [
            results['brute_force']['memory_peak_kb'],
            results['dynamic_programming']['memory_peak_kb'],
            results['dp_space_optimized']['memory_peak_kb']
        ]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'SUBSET SUM Algorithm Comparison - {instance_name}', 
                     fontsize=16, fontweight='bold')
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        # Time Comparison
        bars1 = ax1.bar(algorithms, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Execution Time Comparison', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (bar, time_val) in enumerate(zip(bars1, times)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f} ms',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        bf_time = times[0]
        for i in range(1, 3):
            speedup = bf_time / times[i] if times[i] > 0 else float('inf')
            ax1.text(i, times[i] * 0.5, f'{speedup:.1f}x\nfaster',
                    ha='center', va='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Memory Comparison
        bars2 = ax2.bar(algorithms, memory, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
        ax2.set_title('Peak Memory Usage Comparison', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, mem_val in zip(bars2, memory):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mem_val:.2f} KB',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Time vs Memory
        scatter_sizes = [200, 200, 200]
        for i, (algo, t, m) in enumerate(zip(algorithms, times, memory)):
            ax3.scatter(t, m, s=scatter_sizes[i], color=colors[i], 
                       alpha=0.7, edgecolor='black', linewidth=2, label=algo.replace('\n', ' '))
            ax3.annotate(algo.replace('\n', ' '), (t, m), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))
        
        ax3.set_xlabel('Time (milliseconds)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
        ax3.set_title('Time vs Memory Trade-off', fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        filename = f'subset_sum_comparison_{instance_name.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Diagram saved as: {filename}")
        
        plt.show()


# ========== FSU BENCHMARK LOADER (DYNAMIC DOWNLOAD) ==========
class FSUBenchmarkLoader: 
    BASE_URL = "https://people.sc.fsu.edu/~jburkardt/datasets/subset_sum/"
    CACHE_DIR = "fsu_benchmarks"
    
    # Available benchmark problems
    AVAILABLE_PROBLEMS = {
        'p01': 'P01 - 20 items, large numbers',
        'p02': 'P02 - 6 items, small example',
        'p03': 'P03 - 4 items, tiny example',
        'p04': 'P04 - 9 items, medium',
        'p05': 'P05 - 10 items',
        'p06': 'P06 - 15 items',
        'p07': 'P07 - 20 items (1 to 20)',
        'p08': 'P08 - 24 items'
    }
    
    def __init__(self):
        """Initialize loader and create cache directory."""
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
            print(f"📁 Created cache directory: {self.CACHE_DIR}/")
    
    def download_file(self, filename: str) -> str:
        """
        Download a file from FSU website if not cached.
        
        Args:
            filename: Name of file to download (e.g., 'p01_w.txt')
        
        Returns:
            Local file path
        """
        local_path = os.path.join(self.CACHE_DIR, filename)
        
        # Check if already cached
        if os.path.exists(local_path):
            print(f"  ✓ Using cached: {filename}")
            return local_path
        
        # Download from FSU website
        url = self.BASE_URL + filename
        print(f"  ⬇️  Downloading: {url}")
        
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"  ✓ Downloaded: {filename}")
            return local_path
        except Exception as e:
            raise Exception(f"Failed to download {filename}: {e}")
    
    def parse_file(self, filepath: str) -> List[int]:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract all integers from the file
        numbers = re.findall(r'-?\d+', content)
        return [int(n) for n in numbers]
    
    def load_benchmark(self, problem_id: str) -> Tuple[List[int], int, str]:
        """
        Load a benchmark problem from FSU dataset.
        
        Args:
            problem_id: Problem identifier (e.g., 'p01', 'p02', etc.)
        
        Returns:
            Tuple of (weights, capacity, description)
        """
        problem_id = problem_id.lower()
        
        if problem_id not in self.AVAILABLE_PROBLEMS:
            raise ValueError(f"Unknown problem: {problem_id}. Available: {list(self.AVAILABLE_PROBLEMS.keys())}")
        
        print(f"\n{'='*70}")
        print(f"Loading Benchmark: {problem_id.upper()}")
        print(f"{'='*70}")
        
        # Download files
        weights_file = f"{problem_id}_w.txt"
        capacity_file = f"{problem_id}_c.txt"
        
        weights_path = self.download_file(weights_file)
        capacity_path = self.download_file(capacity_file)
        
        # Parse data
        weights = self.parse_file(weights_path)
        capacity_list = self.parse_file(capacity_path)
        capacity = capacity_list[0] if capacity_list else 0
        
        description = self.AVAILABLE_PROBLEMS[problem_id]
        
        print(f"  ✓ Loaded {len(weights)} weights")
        print(f"  ✓ Target capacity: {capacity}")
        print(f"{'='*70}\n")
        
        return weights, capacity, description
    
    def list_available(self):
        """List all available benchmark problems."""
        print("\n" + "="*70)
        print("AVAILABLE FSU SUBSET SUM BENCHMARKS")
        print("="*70)
        for pid, desc in self.AVAILABLE_PROBLEMS.items():
            print(f"  {pid.upper()}: {desc}")
        print("="*70 + "\n")
    
    def download_all(self):
        """Pre-download all benchmark files for offline use."""
        print("\n" + "="*70)
        print("DOWNLOADING ALL FSU BENCHMARKS")
        print("="*70)
        
        for problem_id in self.AVAILABLE_PROBLEMS.keys():
            try:
                self.load_benchmark(problem_id)
            except Exception as e:
                print(f"  ⚠️  Failed to download {problem_id}: {e}")
        
        print("\n✅ All benchmarks downloaded and cached!")
        print(f"📁 Location: {os.path.abspath(self.CACHE_DIR)}/")
        print("="*70 + "\n")


# ========== COMPREHENSIVE BENCHMARK SUITE ==========
def run_fsu_benchmarks():
    print("\n" + "="*80)
    print("SUBSET SUM - FSU REAL-WORLD BENCHMARK ANALYSIS")
    print("Dataset: Florida State University Subset Sum Collection")
    print("Source: https://people.sc.fsu.edu/~jburkardt/datasets/subset_sum/")
    print("="*80)
    
    # Initialize loader
    loader = FSUBenchmarkLoader()
    loader.list_available()
    
    # Test on selected benchmarks (excluding very large ones for brute force)
    test_problems = ['p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07']
    
    all_results = []
    
    for problem_id in test_problems:
        try:
            # Load benchmark from FSU website
            weights, capacity, description = loader.load_benchmark(problem_id)
            
            print(f"{'='*80}")
            print(f"TESTING: {problem_id.upper()} - {description}")
            print(f"{'='*80}")
            print(f"Items: {len(weights)}")
            print(f"Target: {capacity}")
            print(f"Weights: {weights}")
            
            # Create instance
            ss = SubsetSum(weights, capacity, problem_id.upper())
            
            # Decide whether to run brute force
            if len(weights) > 15:
                print(f"\n⚠️  Skipping brute force (n={len(weights)} > 15)")
                
                # Run only DP
                print("\nRunning Dynamic Programming...")
                tracemalloc.start()
                start = time.perf_counter()
                dp_exists, dp_subset = ss.dynamic_programming()
                dp_time = time.perf_counter() - start
                _, dp_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                print("\nRunning Space-Optimized DP...")
                tracemalloc.start()
                start = time.perf_counter()
                dp_opt_exists, _ = ss.dp_space_optimized()
                dp_opt_time = time.perf_counter() - start
                _, dp_opt_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                print(f"\nResults:")
                print(f"  DP: {dp_time*1000:.3f}ms, Memory: {dp_peak/1024:.2f}KB, Solution: {dp_exists}")
                print(f"  DP-Optimized: {dp_opt_time*1000:.3f}ms, Memory: {dp_opt_peak/1024:.2f}KB, Solution: {dp_opt_exists}")
                
                all_results.append({
                    'problem': problem_id,
                    'description': description,
                    'n': len(weights),
                    'Target': capacity,
                    'dp_time_ms': dp_time * 1000,
                    'dp_memory_kb': dp_peak / 1024,
                    'solution_exists': dp_exists
                })
            else:
                # Run full comparison
                results = ss.compare_algorithms()
                ss.print_results(results)
                ss.plot_comparison(results, problem_id.upper())
                
                all_results.append({
                    'problem': problem_id,
                    'description': description,
                    'n': len(weights),
                    'Target': capacity,
                    'bf_time_ms': results['brute_force']['time_seconds'] * 1000,
                    'dp_time_ms': results['dynamic_programming']['time_seconds'] * 1000,
                    'speedup': results['speedup']['dp_vs_brute_force'],
                    'solution_exists': results['dynamic_programming']['solution_exists']
                })
            
        except Exception as e:
            print(f"\n❌ Error testing {problem_id}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Problem':<10} {'Description':<30} {'n':<6} {'Target':<12} {'DP Time(ms)':<15} {'Solution':<10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['problem']:<10} {r['description'][:28]:<30} {r['n']:<6} {r['Target']:<12} {r['dp_time_ms']:<15.3f} {'Yes' if r['solution_exists'] else 'No':<10}")

    print("\n" + "="*80)
    print("✅ ALL FSU BENCHMARKS COMPLETE!")
    print(f"📁 Data cached in: {os.path.abspath(loader.CACHE_DIR)}/")
    print("="*80)


# ========== MAIN EXECUTION ==========
def main():
    print("\n" + "="*80)
    print("SUBSET SUM PROBLEM - COMPREHENSIVE ANALYSIS")
    print("Using Real FSU Dataset (Automatically Downloaded)")
    print("="*80)
    
    # Run FSU benchmarks
    run_fsu_benchmarks()


if __name__ == "__main__":
    main()