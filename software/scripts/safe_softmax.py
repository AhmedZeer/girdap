import numpy as np

def safe_softmax(input_array):
    # 1. Find the maximum value for numerical stability
    max_val = np.max(input_array)
    
    # 2. Subtract max and exponentiate
    # This ensures the largest value in the array becomes e^0 = 1
    exponents = np.exp(input_array - max_val)
    
    # 3. Calculate the denominator (sum of exponents)
    denominator = np.sum(exponents)
    
    # 4. Normalize to get probabilities
    softmax_output = exponents / denominator
    
    return softmax_output, denominator

# Your provided data
data = np.array([
    0.0, 1.0, 2.0, 3.0, -4.0, -1.5, -2.5, 0.5
])

data = np.array([
    0.0, 1.0, 2.0, 3.0, -4.0, -1.5, -2.5, 0.5,
    0.0, 1.1, 0.0, 0.0,  0.0,  0.0, 3.0 
])


# Execution
new_array, total_sum = safe_softmax(data)

print(f"Denominator (Sum of Exponents): {total_sum:.4f}")
print("New Softmax Array:")
print(new_array)
print(f"\nVerification (Sum of Array): {np.sum(new_array)}")
