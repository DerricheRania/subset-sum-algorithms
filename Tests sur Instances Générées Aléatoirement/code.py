import time
import tracemalloc
from itertools import combinations
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

class SubsetSum:
    """
    Complete implementation of SUBSET SUM problem with:
    - Brute Force approach
    - Dynamic Programming approach
    - Verification function
    - Performance comparison
    - Visual diagrams
    """
    
    def __init__(self, arr: List[int], target: int):
        """
        Initialize with array and target sum.
        
        Args:
            arr: List of positive integers
            target: Target sum to achieve
        """
        self.arr = arr
        self.target = target
        self.n = len(arr)
    
    # ========== BRUTE FORCE APPROACH ==========
    def brute_force(self) -> Tuple[bool, Optional[List[int]]]:
        # Try all possible subset sizes from 1 to n
        for size in range(1, self.n + 1):
            # Generate all combinations of given size
            for subset in combinations(range(self.n), size):
                # Calculate sum of current subset
                current_sum = sum(self.arr[i] for i in subset)
                
                if current_sum == self.target:
                    # Found a solution
                    solution = [self.arr[i] for i in subset]
                    return True, solution
        
        return False, None
    
    # ========== DYNAMIC PROGRAMMING APPROACH ==========
    def dynamic_programming(self) -> Tuple[bool, Optional[List[int]]]:
        # Create DP table
        # dp[i][j] = True if sum j can be achieved using first i elements
        dp = [[False for _ in range(self.target + 1)] for _ in range(self.n + 1)]
        
        # Base case: sum 0 can always be achieved with empty subset
        for i in range(self.n + 1):
            dp[i][0] = True
        
        # Fill the DP table
        for i in range(1, self.n + 1):
            for j in range(1, self.target + 1):
                # Don't include current element
                dp[i][j] = dp[i-1][j]
                
                # Include current element if possible
                if j >= self.arr[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j - self.arr[i-1]]
        
        # Check if solution exists
        if not dp[self.n][self.target]:
            return False, None
        
        # Backtrack to find the actual subset
        subset = []
        i, j = self.n, self.target
        
        while i > 0 and j > 0:
            # If value comes from top (element not included)
            if dp[i-1][j]:
                i -= 1
            # Element was included
            else:
                subset.append(self.arr[i-1])
                j -= self.arr[i-1]
                i -= 1
        
        return True, subset
    
    # ========== OPTIMIZED DP (Space-Optimized) ==========
    def dp_space_optimized(self) -> Tuple[bool, Optional[List[int]]]:
        # dp[j] = True if sum j can be achieved
        dp = [False] * (self.target + 1)
        dp[0] = True
        
        # Process each element
        for num in self.arr:
            # Traverse from right to left to avoid using same element twice
            for j in range(self.target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        
        return dp[self.target], None
    
    # ========== VERIFICATION FUNCTION ==========
    @staticmethod
    def verify_solution(arr: List[int], target: int, subset: Optional[List[int]]) -> bool:
        if subset is None:
            return False
        
        # Check if all elements in subset are in original array
        arr_copy = arr.copy()
        for element in subset:
            if element not in arr_copy:
                return False
            arr_copy.remove(element)  # Remove to handle duplicates correctly
        
        # Check if sum equals target
        return sum(subset) == target
    
    # ========== PERFORMANCE COMPARISON ==========
    def compare_algorithms(self) -> dict:
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
        
        # Calculate speedup
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
            print(f"  Subset: {metrics['subset']}")
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
        
        # Extract data
        times = [
            results['brute_force']['time_seconds'] * 1000,  # Convert to ms
            results['dynamic_programming']['time_seconds'] * 1000,
            results['dp_space_optimized']['time_seconds'] * 1000
        ]
        
        memory = [
            results['brute_force']['memory_peak_kb'],
            results['dynamic_programming']['memory_peak_kb'],
            results['dp_space_optimized']['memory_peak_kb']
        ]
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'SUBSET SUM Algorithm Comparison{" - " + instance_name if instance_name else ""}', 
                     fontsize=16, fontweight='bold')
        
        # Colors for each algorithm
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        # ===== PLOT 1: Time Comparison (Bar Chart) =====
        bars1 = ax1.bar(algorithms, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Execution Time Comparison', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, time_val) in enumerate(zip(bars1, times)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f} ms',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add speedup annotations
        bf_time = times[0]
        for i in range(1, 3):
            speedup = bf_time / times[i] if times[i] > 0 else float('inf')
            ax1.text(i, times[i] * 0.5, f'{speedup:.1f}x\nfaster',
                    ha='center', va='center', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # ===== PLOT 2: Memory Comparison (Bar Chart) =====
        bars2 = ax2.bar(algorithms, memory, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
        ax2.set_title('Peak Memory Usage Comparison', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, mem_val in zip(bars2, memory):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mem_val:.2f} KB',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # ===== PLOT 3: Combined Time-Memory Scatter Plot =====
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
        
        # Save the figure
        filename = f'subset_sum_comparison{"_" + instance_name.replace(" ", "_") if instance_name else ""}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Diagram saved as: {filename}")
        
        plt.show()
    
    @staticmethod
    def plot_scaling_analysis():
        """
        Create a diagram showing how algorithms scale with input size.
        """
        # Simulate scaling for different input sizes
        input_sizes = np.arange(5, 21)
        target = 50  # Fixed target for comparison
        
        bf_times = []
        dp_times = []
        dp_opt_times = []
        
        bf_memory = []
        dp_memory = []
        dp_opt_memory = []
        
        print("\n📈 Running scaling analysis (this may take a moment)...")
        
        for n in input_sizes:
            # Generate random array
            arr = list(np.random.randint(1, 20, size=n))
            
            # Skip if target is impossible to reach
            if sum(arr) < target:
                continue
                
            ss = SubsetSum(arr, target)
            
            # Brute Force
            tracemalloc.start()
            start = time.perf_counter()
            try:
                ss.brute_force()
                bf_times.append((time.perf_counter() - start) * 1000)
                _, peak = tracemalloc.get_traced_memory()
                bf_memory.append(peak / 1024)
            except:
                bf_times.append(None)
                bf_memory.append(None)
            tracemalloc.stop()
            
            # DP
            tracemalloc.start()
            start = time.perf_counter()
            ss.dynamic_programming()
            dp_times.append((time.perf_counter() - start) * 1000)
            _, peak = tracemalloc.get_traced_memory()
            dp_memory.append(peak / 1024)
            tracemalloc.stop()
            
            # Space-Optimized DP
            tracemalloc.start()
            start = time.perf_counter()
            ss.dp_space_optimized()
            dp_opt_times.append((time.perf_counter() - start) * 1000)
            _, peak = tracemalloc.get_traced_memory()
            dp_opt_memory.append(peak / 1024)
            tracemalloc.stop()
            
            print(f"  Completed n={n}")
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Algorithm Scaling Analysis with Input Size', fontsize=16, fontweight='bold')
        
        # ===== Time Scaling =====
        ax1.plot(input_sizes[:len(bf_times)], bf_times, 'o-', color='#e74c3c', 
                linewidth=2, markersize=8, label='Brute Force O(2ⁿ)')
        ax1.plot(input_sizes[:len(dp_times)], dp_times, 's-', color='#3498db', 
                linewidth=2, markersize=8, label='Dynamic Programming O(n×T)')
        ax1.plot(input_sizes[:len(dp_opt_times)], dp_opt_times, '^-', color='#2ecc71', 
                linewidth=2, markersize=8, label='Space-Optimized DP O(n×T)')
        
        ax1.set_xlabel('Input Size (n)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Execution Time vs Input Size', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_yscale('log')  # Log scale to show exponential growth
        
        # ===== Memory Scaling =====
        ax2.plot(input_sizes[:len(bf_memory)], bf_memory, 'o-', color='#e74c3c', 
                linewidth=2, markersize=8, label='Brute Force O(n)')
        ax2.plot(input_sizes[:len(dp_memory)], dp_memory, 's-', color='#3498db', 
                linewidth=2, markersize=8, label='Dynamic Programming O(n×T)')
        ax2.plot(input_sizes[:len(dp_opt_memory)], dp_opt_memory, '^-', color='#2ecc71', 
                linewidth=2, markersize=8, label='Space-Optimized DP O(T)')
        
        ax2.set_xlabel('Input Size (n)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Memory Usage (KB)', fontsize=12, fontweight='bold')
        ax2.set_title('Memory Usage vs Input Size', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('subset_sum_scaling_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Scaling analysis diagram saved as: subset_sum_scaling_analysis.png")
        plt.show()


# ========== DEMONSTRATION AND TESTING ==========
def run_examples(): 
    print("="*70)
    print("SUBSET SUM - COMPLETE ANALYSIS WITH VISUALIZATIONS")
    print("="*70)
    
    # Example 1: Small instance
    print("\n\nEXAMPLE 1: Small instance")
    print("-" * 70)
    arr1 = [3, 34, 4, 12, 5, 2]
    target1 = 9
    print(f"Array: {arr1}")
    print(f"Target: {target1}")
    
    ss1 = SubsetSum(arr1, target1)
    results1 = ss1.compare_algorithms()
    ss1.print_results(results1)
    ss1.plot_comparison(results1, "Small Instance")
    
    # Example 2: No solution exists
    print("\n\nEXAMPLE 2: No solution exists")
    print("-" * 70)
    arr2 = [2, 4, 6, 8]
    target2 = 5
    print(f"Array: {arr2}")
    print(f"Target: {target2}")
    
    ss2 = SubsetSum(arr2, target2)
    results2 = ss2.compare_algorithms()
    ss2.print_results(results2)
    ss2.plot_comparison(results2, "No Solution")
    
    # Example 3: Larger instance
    print("\n\nEXAMPLE 3: Larger instance")
    print("-" * 70)
    arr3 = [1, 5, 11, 5, 3, 7, 19, 23, 29]
    target3 = 42
    print(f"Array: {arr3}")
    print(f"Target: {target3}")
    
    ss3 = SubsetSum(arr3, target3)
    results3 = ss3.compare_algorithms()
    ss3.print_results(results3)
    ss3.plot_comparison(results3, "Large Instance")
    
    # Scaling Analysis
    print("\n\n" + "="*70)
    print("GENERATING SCALING ANALYSIS...")
    print("="*70)
    SubsetSum.plot_scaling_analysis()
    
    print("\n\n" + "="*70)
    print("✅ ALL ANALYSES COMPLETE!")
    print("="*70)
    print("Check the generated PNG files for visualizations:")
    print("  - subset_sum_comparison_Small_Instance.png")
    print("  - subset_sum_comparison_No_Solution.png")
    print("  - subset_sum_comparison_Large_Instance.png")
    print("  - subset_sum_scaling_analysis.png")


if __name__ == "__main__":
    run_examples()