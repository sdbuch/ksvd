"""
Implementation of the Matching, Stretching, and Projection (MSP) algorithm for orthogonal dictionary learning.

This implements the algorithm from [ZMZM20] which solves the optimization problem:
    min (1/4)||AX||_4^4  subject to  A^T A = I

The algorithm uses power iteration with orthogonal projection to find the optimal orthogonal dictionary.
"""

# Configure JAX to use double precision
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional
from functools import partial
import matplotlib.pyplot as plt
import torch
import torchvision
import os
from pathlib import Path


def project_orthogonal(A: jnp.ndarray) -> jnp.ndarray:
    """Project a matrix onto the space of orthogonal matrices O(D) using SVD.
    
    Args:
        A: Input matrix of shape (D, D)
        
    Returns:
        Projected orthogonal matrix of shape (D, D)
    """
    U, _, Vh = jnp.linalg.svd(A, full_matrices=True)
    return U @ Vh


@partial(jax.jit, static_argnums=(3, 4))
def msp_iteration(A: jnp.ndarray, 
                 X: jnp.ndarray, 
                 key: jnp.ndarray,
                 max_iters: int = 100,
                 tol: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the MSP algorithm to learn an orthogonal dictionary.
    
    Args:
        A: Initial orthogonal dictionary matrix of shape (D, D)
        X: Data matrix of shape (D, N) where N is number of samples
        key: JAX random key for reproducibility
        max_iters: Maximum number of iterations
        tol: Convergence tolerance on the Frobenius norm of successive iterates
        
    Returns:
        Tuple of:
            - Learned orthogonal dictionary matrix
            - Array of errors (Frobenius norm of successive iterates)
            - Array of objective values
    """
    def body_fun(A_prev, _):
        # MSP update step: A_{t+1} = P_O(D)[(A_t X)^⊙3 X^T]
        AX = A_prev @ X
        AX_cubed = AX ** 3  # Element-wise cube
        update = AX_cubed @ X.T
        A_next = project_orthogonal(update)
        
        # Compute error and objective value
        error = jnp.linalg.norm(A_next - A_prev, ord='fro')
        obj_value = 0.25 * jnp.sum(jnp.power(A_next @ X, 4))
        
        return A_next, (error, obj_value)

    A_final, (errors, obj_values) = jax.lax.scan(
        body_fun, A, None, length=max_iters)
    
    return A_final, errors, obj_values


def initialize_orthogonal(D: int, key: jnp.ndarray) -> jnp.ndarray:
    """Initialize a random orthogonal matrix using QR decomposition.
    
    Args:
        D: Dimension of the square matrix
        key: JAX random key
        
    Returns:
        Random orthogonal matrix of shape (D, D)
    """
    key1, key2 = random.split(key)
    A = random.normal(key1, (D, D))
    Q, _ = jnp.linalg.qr(A)
    return Q


def generate_sparse_data(D: int, N: int, K: int, key: jnp.ndarray) -> jnp.ndarray:
    """Generate a sparse data matrix where each column has K nonzero Gaussian entries.
    
    Args:
        D: Dimension of each vector (column)
        N: Number of vectors (columns)
        K: Number of nonzero entries per column
        key: JAX random key
        
    Returns:
        Sparse matrix of shape (D, N) with K nonzero entries per column
    """
    if K > D:
        raise ValueError(f"Cannot have more nonzeros (K={K}) than dimensions (D={D})")
        
    key1, key2 = random.split(key)
    
    # Generate N sets of K unique indices from 0 to D-1
    # We do this by generating one set at a time to avoid the N*K > D issue
    indices = jnp.zeros((N, K), dtype=jnp.int32)
    for i in range(N):
        key1, subkey = random.split(key1)
        indices = indices.at[i].set(random.choice(subkey, D, shape=(K,), replace=False))
    
    # Generate Gaussian values for nonzero entries
    values = random.normal(key2, shape=(N, K))
    
    # Create sparse matrix by scattering values to their positions
    X = jnp.zeros((N, D))
    X = X.at[jnp.arange(N)[:, None], indices].set(values)
    
    # Transpose and normalize columns
    X = X.T
    X = X / jnp.linalg.norm(X, axis=0)
    
    return X


def load_mnist_data(num_samples: int = 1000, key: jnp.ndarray = None) -> jnp.ndarray:
    """Load MNIST digits and convert to normalized column vectors.
    
    Args:
        num_samples: Number of digits to load
        key: JAX random key for shuffling
        
    Returns:
        Matrix of shape (784, num_samples) where each column is a normalized MNIST digit
    """
    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Download and load MNIST
    mnist = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    # Convert to array and flatten images
    X = torch.stack([img.reshape(-1) for img, _ in mnist])  # Shape: (60000, 784)
    X = X.numpy()
    
    # Convert to JAX array
    X = jnp.array(X)
    
    # Randomly select num_samples if key provided
    if key is not None and num_samples < len(X):
        indices = random.choice(key, len(X), shape=(num_samples,), replace=False)
        X = X[indices]
    else:
        X = X[:num_samples]
    
    # Transpose to get (784, num_samples) and normalize columns
    X = X.T
    # X = X / jnp.linalg.norm(X, axis=0)

    # breakpoint()
    
    return X


def visualize_dictionary(A: jnp.ndarray, nrows: int = 2, ncols: int = 5, figsize: Tuple[int, int] = (15, 6)):
    """Visualize learned dictionary atoms as MNIST-sized images.
    
    Args:
        A: Dictionary matrix of shape (784, D)
        nrows: Number of rows in the plot
        ncols: Number of columns in the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel()
    
    for i in range(min(nrows * ncols, A.shape[1])):
        # Reshape atom to 28x28 and normalize for visualization
        atom = A[:, i].reshape(28, 28)
        vmax = jnp.abs(atom).max()
        axes[i].imshow(atom, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    # Parameters
    N = 10000  # Number of samples
    D = 10    # Dictionary dimension
    
    # Initialize random key
    key = random.PRNGKey(0)
    key1, key2, key3 = random.split(key, 3)
    
    # Choose which data to use (sparse or MNIST)
    use_mnist = True
    
    if use_mnist:
        X = load_mnist_data(N, key1)
        print(f"\nLoaded MNIST data matrix of shape: {X.shape}")
    else:
        K = 3  # Number of nonzero entries per column
        X = generate_sparse_data(D, N, K, key1)
        print(f"\nGenerated sparse data with {K} nonzeros per column")
    
    # Get problem dimension from data
    D = X.shape[0]
    
    # Initialize random orthogonal dictionary
    A_init = initialize_orthogonal(D, key2)
    
    # Run MSP algorithm
    A_learned, errors, obj_values = msp_iteration(A_init, X, key3)
    
    # Create figure with subplots
    if use_mnist:
        # For MNIST, show convergence plots and dictionary atoms
        fig = plt.figure(figsize=(15, 10))
        
        # Plot error convergence
        ax1 = plt.subplot(221)
        iterations = jnp.arange(len(errors))
        ax1.semilogy(iterations, errors)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Error (Frobenius norm)')
        ax1.set_title('Convergence of Successive Iterates')
        ax1.grid(True)
        
        # Plot objective value
        ax2 = plt.subplot(222)
        ax2.plot(iterations, obj_values)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('Objective Value vs Iteration')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Show learned dictionary atoms
        print("\nLearned dictionary atoms:")
        fig = visualize_dictionary(A_learned.T)
        plt.show()
        
    else:
        # Original visualization for sparse case
        # Create figure with three subplots
        fig = plt.figure(figsize=(15, 5))
        
        # Plot error convergence
        ax1 = plt.subplot(131)
        iterations = jnp.arange(len(errors))
        ax1.semilogy(iterations, errors)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Error (Frobenius norm)')
        ax1.set_title('Convergence of Successive Iterates')
        ax1.grid(True)
        
        # Plot objective value
        ax2 = plt.subplot(132)
        ax2.plot(iterations, obj_values)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('Objective Value vs Iteration')
        ax2.grid(True)
        
        # Plot A_learned to check for signed permutation of I
        ax3 = plt.subplot(133)
        im = ax3.imshow(A_learned, cmap='RdBu', vmin=-1, vmax=1)
        ax3.set_title('A_learned\n(Should be signed permutation of I)')
        plt.colorbar(im, ax=ax3)
        
        plt.tight_layout()
        plt.show()
    
    # Print final metrics
    print(f"\nFinal error between successive iterates: {errors[-1]:.6e}")
    print(f"Final objective value: {obj_values[-1]:.6f}")
    
    # Verify orthogonality of learned matrix
    error_ortho = jnp.linalg.norm(A_learned.T @ A_learned - jnp.eye(D), ord='fro')
    print(f"Orthogonality error: {error_ortho:.6e}")
    
    if not use_mnist:
        # Only show permutation metrics for sparse case
        # For MNIST we don't expect to recover I
        max_entries = jnp.maximum(jnp.abs(A_learned).max(axis=0), jnp.abs(A_learned).max(axis=1))
        min_max_entry = jnp.min(max_entries)
        print(f"\nMin of max absolute entries per row/col: {min_max_entry:.6f} (should be close to 1)")
        
        close_to_one = jnp.sum(jnp.abs(A_learned) > 0.9)
        print(f"Number of entries with |value| > 0.9: {close_to_one} (should be {D} for perfect recovery)")
        
        rows_good = jnp.sum(jnp.sum(jnp.abs(A_learned) > 0.9, axis=1) == 1) == D
        cols_good = jnp.sum(jnp.sum(jnp.abs(A_learned) > 0.9, axis=0) == 1) == D
        print(f"Each row has exactly one |value| > 0.9: {rows_good}")
        print(f"Each col has exactly one |value| > 0.9: {cols_good}") 