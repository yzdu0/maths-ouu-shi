"""Finite checks for the signed-attention and Gaussian reductions. Run with Python 3.

If h = b**2/(2*d**2), every constructed logit is an integer multiple of h.
The variable u stands for exp(h). The identity in u is checked symbolically;
the normalized rows are also evaluated at rational u > 1 using Fraction.
The separate Gaussian check uses exact squared distances and 100-digit exponentials.
"""

from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement, product
from math import comb, isqrt, lcm, prod


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def coefficients(d):
    """Return the nonzero coefficients of the monomials z**t * u**r."""
    terms = {(0, 0): 1}
    for j in range(1, d + 1):
        updated = Counter()
        for (t, r), coefficient in terms.items():
            updated[t, r + j] += coefficient
            updated[t + 1, r] -= coefficient
        terms = {key: value for key, value in updated.items() if value}
    return terms


def check_polynomial(d):
    terms = coefficients(d)
    subset_terms = Counter()
    for mask in range(1 << d):
        factors = [j for j in range(1, d + 1) if mask & (1 << (j - 1))]
        t = d - len(factors)
        subset_terms[t, sum(factors)] += (-1) ** t
    require(terms == subset_terms, ("subset expansion", d))
    require(sum(map(abs, terms.values())) == 2**d, ("coefficient mass", d))
    zero_terms = Counter()
    for mask in range(1 << d):
        factors = [j for j in range(1, d + 1) if mask & (1 << (j - 1))]
        zero_terms[sum(factors)] += (-1) ** (d - len(factors))
    for overlap in range(d + 1):
        specialized = Counter()
        for (t, r), coefficient in terms.items():
            specialized[t * overlap + r] += coefficient
        expected = zero_terms if overlap == 0 else {}
        nonzero = {key: value for key, value in specialized.items() if value}
        reference = {key: value for key, value in expected.items() if value}
        require(nonzero == reference, ("formal identity", d, overlap))
    return d + 1


def binary_vectors(width):
    return list(product((0, 1), repeat=width))


def padded_vector(vector, d):
    return tuple(vector) + (0,) * (d - len(vector))


def dot(vector, other):
    require(len(vector) == len(other), "dimension mismatch")
    return sum(x * y for x, y in zip(vector, other))


def check_instance(xs, ys, bound, dummy_weight, base):
    width = len(xs[0])
    d = 1 << (width - 1).bit_length()
    h = bound**2 / (2 * d**2)
    terms = coefficients(d)
    zero_key = (0,) * width
    blocks = [(y, Fraction(1)) for y in ys] + [(zero_key, dummy_weight)]
    entries = []
    for y, scale in blocks:
        y_padded = padded_vector(y, d)
        for (t, r), coefficient in terms.items():
            key = tuple(bound * t * bit / d for bit in y_padded)
            key += (bound * r / d**2,) * d
            value = scale * bound * coefficient / 2**d
            require(max(map(abs, key)) <= bound and abs(value) <= bound, "entry bound")
            entries.append((key, value, y, t, r))

    require(len(entries) >= len(xs), "square padding")
    padded_xs = list(xs) + [xs[0]] * (len(entries) - len(xs))
    require(len(padded_xs) == len(entries), "square matrix size")
    require(all(x == xs[0] for x in padded_xs[len(xs):]), "padded queries")
    row_results = {}
    scores_checked = 0
    scale_at_zero = bound * prod(base**j - 1 for j in range(1, d + 1)) / 2**d
    for x in dict.fromkeys(padded_xs):
        query = tuple(bound * bit for bit in padded_vector(x, d)) + (bound,) * d
        require(max(map(abs, query)) <= bound, "query bound")
        numerator = denominator = Fraction(0)
        for key, value, y, t, r in entries:
            exponent = dot(query, key) / (2 * d) / h
            require(exponent == t * dot(x, y) + r, "constructed dot product")
            require(exponent.denominator == 1, "integral exponent")
            weight = base ** int(exponent)
            numerator += weight * value
            denominator += weight
            scores_checked += 1
        orthogonal_count = sum(dot(x, y) == 0 for y in ys)
        expected = scale_at_zero * (orthogonal_count + dummy_weight)
        require(numerator == expected, "numerator identity")
        require(denominator > 0, "denominator")
        row_results[x] = numerator / denominator
        require(row_results[x] == expected / denominator, "normalized output")
        if dummy_weight < 0:
            require(row_results[x] != 0, "nonzero signed output")
            require((row_results[x] > 0) == (orthogonal_count > 0), "sign decision")
        else:
            require(row_results[x] > 0, "strictly positive output")

    # Rows padded by repeated queries reuse exactly the same coordinates and values.
    normalized = [row_results[x] for x in padded_xs]
    require(all(y == normalized[0] for y in normalized[len(xs):]), "padded outputs")
    return scores_checked, len(row_results)


@lru_cache(maxsize=None)
def four_squares(number):
    """Find an integer four-square representation by joining two sums of squares."""
    pairs = {}
    for i in range(isqrt(number) + 1):
        for j in range(isqrt(number - i * i) + 1):
            pairs.setdefault(i * i + j * j, (i, j))
    for subtotal, pair in pairs.items():
        if number - subtotal in pairs:
            return pair + pairs[number - subtotal]
    raise AssertionError(("missing four-square representation", number))


def equal_norm_lift(y, t, r):
    d = len(y)
    base = tuple(Fraction(t * bit, d) for bit in y) + (Fraction(r, d**2),) * d
    residual = (2 * d - dot(base, base)) * d**4
    require(residual.denominator == 1 and residual >= 0, "integral lift residual")
    lift = tuple(Fraction(k, d**2) for k in four_squares(int(residual)))
    z = base + lift
    require(dot(z, z) == 2 * d, "common lifted key norm")
    return z


def decimal_of(number):
    return Decimal(number.numerator) / Decimal(number.denominator)


def check_gaussian_gram():
    """Evaluate the full symmetric Gaussian matvec at 100 decimal digits."""
    d = 2
    n = 2
    terms = coefficients(d)
    m = n + 2 * n * len(terms)
    tau = Fraction(1, 2 ** (8 * d + 3 * (m - 1).bit_length()))
    all_inputs = list(combinations_with_replacement(binary_vectors(d), n))
    blocks = {}
    for y in binary_vectors(d):
        block = []
        for (t, r), coefficient in terms.items():
            z = equal_norm_lift(y, t, r)
            for sign in (-1, 1):
                block.append((z + (sign * tau,), Fraction(sign * coefficient, 2**d)))
        blocks[y] = block

    @lru_cache(maxsize=None)
    def gaussian(p, q):
        squared_distance = sum((a - b) ** 2 for a, b in zip(p, q))
        return (-decimal_of(squared_distance)).exp()

    counts = {False: 0, True: 0}
    with localcontext() as context:
        context.prec = 100
        no_upper = decimal_of(4 * tau**2 * n) * Decimal(m).sqrt()
        yes_lower = decimal_of(tau / (2 ** (8 * d + 1)))
        for xs in all_inputs:
            query_rows = [(x + (1,) * d + (0,) * 4 + (1,), 0) for x in xs]
            for ys in all_inputs:
                entries = query_rows + [entry for y in ys for entry in blocks[y]]
                require(len(entries) == m, "Gaussian Gram size")
                # Compute the matrix product directly from pairwise distances.
                product_rows = [
                    sum(gaussian(p, q) * decimal_of(weight) for q, weight in entries)
                    for p, _ in entries
                ]
                for x, value in zip(xs, product_rows):
                    have_partner = any(dot(x, y) == 0 for y in ys)
                    if have_partner:
                        require(value >= yes_lower, "Gaussian witness coordinate")
                    else:
                        require(abs(value) < Decimal("1e-85"), "Gaussian query cancellation")
                norm = sum(value * value for value in product_rows).sqrt()
                have_pair = any(dot(x, y) == 0 for x in xs for y in ys)
                counts[have_pair] += 1
                if have_pair:
                    require(norm >= yes_lower, "Gaussian YES bound")
                else:
                    require(norm <= no_upper, "Gaussian NO bound")
    return counts[False], counts[True], m


def log_filter(k):
    """Integer coefficients of lcm(1,...,k) times the Taylor polynomial of log at 1."""
    scale = lcm(*range(1, k + 1))
    terms = Counter()
    for ell in range(1, k + 1):
        multiplier = (-1) ** (ell + 1) * (scale // ell)
        for j in range(ell + 1):
            terms[j] += multiplier * comb(ell, j) * (-1) ** (ell - j)
    for ell in range(k + 1):
        coefficient_at_one = sum(value * comb(j, ell) for j, value in terms.items()
                                 if j >= ell)
        expected = (-1) ** (ell + 1) * (scale // ell) if ell else 0
        require(coefficient_at_one == expected, "exact logarithm Taylor coefficients")
    return scale, {j: value for j, value in terms.items() if value}


def check_binary_values(d):
    """Group identical +/-1 keys by explicit integer multiplicity."""
    k = (d + 1).bit_length() ** 2  # square of ceil(log2(d+2))
    log_scale, log_terms = log_filter(k)
    terms = coefficients(d)
    dimension = 2 * d + 1
    h = Fraction(1, dimension * d)
    theta = Fraction(1, dimension * 2**d)
    ones, zeros = (1,) * d, (0,) * d
    alternating = tuple(i % 2 for i in range(d))
    first = (1,) + (0,) * (d - 1)
    xs = (zeros, ones, alternating, first)
    ys = (ones, first, first, alternating)

    @lru_cache(maxsize=None)
    def exp_exact_argument(argument):
        return decimal_of(argument).exp()

    groups = []
    original_blocks = [(y, Fraction(1)) for y in ys] + [(zeros, Fraction(-1, 2))]
    for y, scale in original_blocks:
        for (t, r), c in terms.items():
            value = scale * Fraction(c, 2**d)
            w = abs(value)
            original_key = tuple(Fraction(t * bit, d) for bit in y) + (Fraction(r, d**2),) * d
            for j, coefficient in log_terms.items():
                key = original_key + (j * w / 2**d,)
                sign = (1 if value > 0 else -1) * (1 if coefficient > 0 else -1)
                require(sign in (-1, 1) and max(map(abs, key)) <= 1, "binary key bound")
                groups.append((key, abs(coefficient), sign, y, t, r, j, w))

    checks = positive = negative = 0
    with localcontext() as context:
        context.prec = 170
        target_scale = log_scale * decimal_of(theta)
        p0 = prod(exp_exact_argument(h * i) - 1 for i in range(1, d + 1))
        for x in xs:
            query = x + ones + (1,)
            numerator = denominator = error_bound = Decimal(0)
            for key, copies, sign, y, t, r, j, w in groups:
                logit = dot(query, key) / dimension
                require(logit == h * (t * dot(x, y) + r) + theta * j * w,
                        "binary coordinate-derived logit")
                mass = copies * exp_exact_argument(logit)
                numerator += sign * mass
                denominator += mass
                checks += 1
            for y, scale in original_blocks:
                for (t, r), c in terms.items():
                    w = abs(scale * Fraction(c, 2**d))
                    increment = exp_exact_argument(theta * w) - 1
                    require(0 < increment < 1, "logarithm series domain")
                    remainder = log_scale * increment ** (k + 1) / (k + 1)
                    error_bound += exp_exact_argument(h * (t * dot(x, y) + r)) * remainder
            partners = sum(dot(x, y) == 0 for y in ys)
            target = target_scale * p0 * (Decimal(partners) - Decimal("0.5")) / 2**d
            require(abs(numerator - target) <= error_bound + Decimal("1e-140"),
                    "binary numerator Taylor error")
            require(error_bound < abs(target) / 2, "binary sign gap")
            require((numerator / denominator > 0) == (partners > 0), "binary output sign")
            positive += int(partners > 0)
            negative += int(partners == 0)
    return checks, positive, negative


def main():
    formal_cases = sum(check_polynomial(d) for d in (1, 2, 4, 8))
    instances = []
    for width in range(1, 5):
        all_bits = binary_vectors(width)
        ones = all_bits[-1]
        instances.append((all_bits, [ones] * len(all_bits)))
        instances.append((all_bits, all_bits))
        keys = [all_bits[(3 * i + 1) % len(all_bits)] for i in range(len(all_bits))]
        instances.append((all_bits, keys))
    five_bits = binary_vectors(5)
    instances.append(([five_bits[i] for i in (0, 3, 15, 31)],
                      [five_bits[i] for i in (3, 3, 12, 31)]))

    scores = rows = cases = 0
    for index, (xs, ys) in enumerate(instances):
        bound = Fraction(1, 2 ** (1 + index % 6))
        base = (Fraction(3, 2), Fraction(17, 16))[index % 2]
        for dummy_weight in (Fraction(-1, 2), Fraction(1, 16)):
            tested_scores, tested_rows = check_instance(xs, ys, bound, dummy_weight, base)
            scores += tested_scores
            rows += tested_rows
            cases += 1
    print(f"Passed {formal_cases} formal polynomial specializations (d = 1, 2, 4, 8).")
    print(f"Passed {cases} assembled square instances, {rows} distinct-query row checks, "
          f"and {scores} exact logit checks.")
    print("Both dummy signs, repeated keys, non-power-of-two widths, and padding passed.")
    no_cases, yes_cases, m = check_gaussian_gram()
    print(f"Passed equal-norm lifts and complete {m}-by-{m} Gaussian matrix-vector checks "
          f"for {no_cases} NO and {yes_cases} YES instances (d = 2, 100 decimal digits).")
    binary_checks = [check_binary_values(d) for d in (4, 8)]
    n_logits, n_positive, n_negative = map(sum, zip(*binary_checks))
    print(f"Passed binary-value filters at degrees 9 and 16, {n_logits} grouped logit "
          f"checks, and Taylor error bounds for {n_positive} positive and {n_negative} "
          "negative outputs (170 decimal digits).")


if __name__ == "__main__":
    main()
