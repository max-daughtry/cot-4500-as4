# Max Daughtry
# Apr 11, 2024
# COT 4500
# Assignment 4

import numpy as np

# Problem 1: Jacobi Iterative Method
def JacobiIterative():
    # Set the tolerance
    TOL = 10 ** -3

    # Set the augmented matrix A|b
    A = [[10, -1, 2, 0, 6], [-1, 11, -1, 3, 25], [2, -1, 10, -1, -11], [0, 3, -1, 8, 15]]
    # Dimension of matrix A
    n = len(A)

    # Previous x approximation (x^(k-1))
    XO = np.array([0] * n, dtype=float)

    # Maximum number of iterations
    N = 500

    # Current number of iterations
    k = 1

    # Current x approximation (x^(k))
    x = np.array([0] * n, dtype=float)

    # Loop until maximum iterations met
    while k <= N:
        # Loop over each row of A
        for i in range(n):
            # Create sum value
            sum = 0
            # Loop over each column of A
            for j in range(n):
                if j != i:
                    sum += A[i][j] * XO[j]

            # Calculate x^(k) for each index i
            x[i] = (1 / A[i][i]) * (A[i][len(A)] - sum)

        # If the difference between x^(k) and x^(k-1) is within the tolerance,
        # print result and break program
        if np.linalg.norm(x-XO) < TOL:
            print(x)
            return
            
        # Increment current number of iterations
        k += 1

        # Set x^(k-1) to x^(k)
        for i in range(n):
            XO[i] = x[i]
    
    print('Maximum number of iterations exceeded')
    return

# Problem 2: Gauss Seidel Iterative Method
def GaussSeidelIterative():
    # Set the tolerance
    TOL = 10 ** -3

    # Set the augmented matrix A|b
    A = [[10, -1, 2, 0, 6], [-1, 11, -1, 3, 25], [2, -1, 10, -1, -11], [0, 3, -1, 8, 15]]
    # Set the dimension of A
    n = len(A)

    # Set max number of iterations
    N = 100

    # Set current number of iterations
    k = 1
    
    # Set x^(k-1) and x^(k) respectively where x^(k) is the initial approximation
    XO = np.array([0] * n, dtype=float)
    x = np.array([0] * n, dtype=float)

    # Loop until maximum iterations met
    while k <= N:
        # Loop over each row of A
        for i in range(n):
            # Initialize sum value
            sum = 0

            # Loops below ensure j != i

            # Loop from 0 (inclusive) to i (exclusive)
            for j in range(i):
                sum += A[i][j] * x[j]
            # Loop from i+1 (inclusive) to n (exclusive)
            for j in range(i+1, n):
                sum += A[i][j] * XO[j]

            # Calculate x^(k) for each index i
            x[i] = (1 / A[i][i]) * (A[i][len(A)] - sum)

        # If the difference between x^(k) and x^(k-1) is within the tolerance,
        # print result and break program
        if np.linalg.norm(x-XO) < TOL:
            print(x)
            return
        
        # Increment current number of iterations
        k += 1

        # Set x^(k-1) to x^(k)
        for i in range(n):
            XO[i] = x[i]
    
    print('Maximum number of iterations exceeded')
    return

# Problem 3: SOR Method
def SORMethod():
    # Set tolerance
    TOL = 10 ** -3

    # Set w value
    w = 1.25

    # Set matrix A
    A = [[4, 3, 0, 24], [3, 4, -1, 30], [0, -1, 4, -24]]
    # Dimension of matrix A
    n = len(A)

    # Set max number of iterations
    N = 100

    # Current number of iterations
    k = 1
    
    # Set x^(k-1) and x^(k) respectively where x^(k) is the initial approximation
    XO = np.array([0] * n, dtype=float)
    x = np.array([1] * n, dtype=float)

    # Loop until maximum iterations met
    while k <= N:
        # Loop over each row of A
        for i in range(n):
            # Initialize sum value
            sum = 0
            
            # Loops below ensure j != i

            # Loop from 0 (inclusive) to i (exclusive)
            for j in range(i):
                sum += A[i][j] * x[j]
            
            # Loop from i+1 (inclusive) to n (exclusive)
            for j in range(i+1, n):
                sum += A[i][j] * XO[j]

            # Calculate x^(k) for each index i
            x[i] = (1 / A[i][i]) * w * (A[i][len(A)] - sum) + ((1-w) * XO[i])

        # If the difference between x^(k) and x^(k-1) is within the tolerance,
        # print result and break program
        if np.linalg.norm(x-XO) < TOL:
            print(x)
            return
            
        # Increment current number of iterations
        k += 1

        # Set x^(k-1) to x^(k)
        for i in range(n):
            XO[i] = x[i]
    
    print('Maximum number of iterations exceeded')
    return

# Problem 4: Iterative Refinement Method
def IterativeRefinementMethod():
    # Set tolerance
    TOL = 10**-5
    # Set max number of iterations
    N = 100
    # Initialize COND
    COND = None
    # Set t for t-arithmetic
    t = 5

    # Set matrix A
    A = [[3.3330, 15920, -10.333], [2.2220, 16.710, 9.6120], [1.5611, 5.1791, 1.6852]]
    # Set vector B
    b = np.array([15913, 28.544, 8.4254], dtype=float)
    # Dimension of matrix A
    n = len(A)

    # Augmented matrix A|b
    Ab = np.array([[3.3330, 15920, -10.333, 15913], [2.2220, 16.710, 9.6120, 28.544], [1.5611, 5.1791, 1.6852, 8.4254]], dtype=float)

    # Current number of iterations
    k = 1

    # r and xx vectors
    r = np.zeros(n, dtype=float)
    xx = np.zeros(n, dtype=float)

    # Loop until max number of iterations reached
    while k <= N:
        # Solve system of equations using Gaussian elimination
        x = SolveMatrix(Ab)
        
        # Loop over each row of A
        for i in range(n):
            # Initialize sum value
            sum = 0

            # Loop from 0 (inclusive) to i (exclusive)
            for j in range(n):
                sum += A[i][j] * x[j]
            
            # Set r[i] for each index i
            r[i] = b[i] - sum

        # Create augmented matrix A|r
        Ar = np.zeros((n, n+1), dtype=float)
        for i in range(len(A)):
            for j in range(len(A[i])):
                Ar[i][j] = A[i][j]
            Ar[i][n] = r[i]

        # Solve system of equations using Gaussian elimination
        y = SolveMatrix(Ar)

        # Loop over each row of A
        for i in range(n):
            # Set xx[i] for each index i
            xx[i] = x[i] + y[i]

        # If in the first iteration set COND value
        if k == 1:
            COND = (np.linalg.norm(y) / np.linalg.norm(xx)) * (10 ** t)

        # If x-xx is within the tolerance print xx and COND and break program
        if np.linalg.norm(x-xx) < TOL:
            print(xx)
            print(COND)
            return
        
        # Increment current number of iterations
        k += 1

        # Set x^(k-1) to x^(k)
        for i in range(n):
            x[i] = xx[i]

    return

# Used for solving system of equations for some augmented matrix A
def SolveMatrix(A):
    # Dimension of matrix A
    n = len(A)

    # Gaussian elimination
    for i in range(1, n):
        p = i

        if not (i <= p <= n and A[p-1][i-1] != 0):
            print("No unique solution exists...")
            return

        if (p != i):
            swap = A[i-1]
            A[i-1] = A[p-1]
            A[p-1] = swap
        
        for j in range(i+1, n+1):
            m = A[j-1][i-1] / A[i-1][i-1]
            A[j-1] = A[j-1] - m * A[i-1]
        
    if A[n-1][n-1] == 0:
        print("No unique solution exists...")
        return
    
    # Back substitution to find solution x

    x = np.zeros(n)

    x[n-1] = A[n-1][n] / A[n-1][n-1]

    for i in range(n-1, 0, -1):
        sumTerm = 0
        for j in range(i+1, n+1):
            sumTerm += A[i-1][j-1]*x[j-1]
        x[i-1] = (A[i-1][n] - sumTerm)/A[i-1][i-1]
    
    return x

# Run all functions
if __name__ == '__main__':
    JacobiIterative()
    print()
    print()
    GaussSeidelIterative()
    print()
    print()
    SORMethod()
    print()
    print()
    IterativeRefinementMethod()
