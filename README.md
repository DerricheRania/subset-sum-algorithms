## 🧮 Subset Sum – Algorithmic Study & Experimental Evaluation

This project presents a complete **algorithmic and experimental study of the Subset Sum problem**, a classical **NP-Complete decision problem**. The goal is to determine whether a subset of a given set of positive integers sums exactly to a target value, and optionally retrieve that subset.

## 🚀 Implemented Approaches

Three algorithmic strategies were implemented and analyzed:

### 1️⃣ Brute Force (Exhaustive Search)

* Explores all possible subsets (2ⁿ)
* Guarantees a solution if it exists
* Practical only for small instances (n ≤ 20)

### 2️⃣ Dynamic Programming (2D Table)

* Uses a boolean DP table `dp[i][j]` to track reachable sums
* Efficient for moderate target values
* Allows **reconstruction of the solution subset**

### 3️⃣ Space-Optimized Dynamic Programming (1D Table)

* Reduces memory from O(n×T) to O(T)
* Maintains the same time complexity
* Does **not** support subset reconstruction

## ⏱️ Complexity Overview

| Approach            | Time Complexity | Space Complexity | Subset Reconstruction |
| ------------------- | --------------- | ---------------- | --------------------- |
| Brute Force         | O(2ⁿ · n)       | O(n)             | ✅ Yes                 |
| Dynamic Programming | O(n · T)        | O(n · T)         | ✅ Yes                 |
| Optimized DP        | O(n · T)        | O(T)             | ❌ No                  |

## 🧪 Experimental Evaluation

The algorithms were evaluated using **two complementary testing strategies**:

### 🔹 Randomly Generated Instances

* Controlled problem sizes (n = 5 to 30)
* Fixed target value for fair comparison
* Used to study scalability, runtime growth, and correctness

### 🔹 Standard Academic Benchmarks (FSU Dataset)

* Real-world instances from the **Florida State University Subset Sum Collection**
* Benchmarks P01–P07, including large-capacity cases
* Results validated against known published solutions

## 📊 Key Findings

* Brute force shows **exponential growth** and becomes infeasible beyond n ≈ 20
* Dynamic Programming scales **linearly with n** (for fixed T)
* Performance is highly sensitive to the **magnitude of the target value T**
* Large-capacity instances (e.g. P03) confirm the **pseudo-polynomial nature** of DP
* Space-optimized DP significantly reduces memory usage with minimal trade-offs

## 📈 Visualizations

The project automatically generates plots illustrating:

* Runtime comparison between algorithms
* Memory usage analysis
* Scalability trends (logarithmic scale)
* Time vs memory trade-offs

## ✅ Validation & Testing

* All returned subsets are verified for correctness
* Handles edge cases (no solution, empty set, target = 0, duplicates)
* Tested on both synthetic and real benchmark datasets
* Results exported for reproducibility and analysis

## 🎯 Project Goal

This repository aims to provide a **clear, practical, and experimentally validated comparison** of Subset Sum algorithms, bridging **theoretical complexity analysis** with **real-world performance measurements**.
