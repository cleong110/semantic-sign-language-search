import math
import argparse



def hypergeometric_probability(k, N, K, n):
    """
    Calculate the hypergeometric probability P(X = k) using Python's standard library.
    """
    return (math.comb(K, k) * math.comb(N - K, n - k)) / math.comb(N, n)

def expected_correct_results(N, K, n):
    """
    Calculate the expected number of correct results without using numpy.
    """
    expected_value = sum(k * hypergeometric_probability(k, N, K, n) for k in range(n + 1))
    return expected_value

def main():
    parser = argparse.ArgumentParser(description="Calculate the expected number of correct search results.")
    parser.add_argument('-N', type=int, required=True, help="Total population size")
    parser.add_argument('-K', type=int, required=True, help="Number of correct search results in the population")
    parser.add_argument('-n', type=int, required=True, help="Number of search results retrieved")

    args = parser.parse_args()

    expected_value = expected_correct_results(args.N, args.K, args.n)
    print(f"Expected number of correct results: {expected_value:.4f}")

if __name__ == "__main__":
    main()

