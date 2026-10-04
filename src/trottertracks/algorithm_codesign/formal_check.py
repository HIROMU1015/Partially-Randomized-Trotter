"""Formula-only noncommutative word-series residuals, no matrix/signal input."""
import math
import mpmath as mp


def order_residuals(point, degree=8):
    """Compare S2 composition to exp(t(A+B)) through the requested degree.

    Two free noncommuting generators catch mixed-word errors in addition
    to power sums. This is a coefficient-transcription check, not a proof
    that finite RTE is an eighth-order algorithm.
    """
    with mp.workdps(80):
        coefficients = {'': mp.mpf(1)}
        for weight in point.weights:
            w = weight.value(point.basis)
            for letter, time in (('A', w/2), ('B', w), ('A', w/2)):
                updated = {}
                for word, value in coefficients.items():
                    for n in range(degree-len(word)+1):
                        new_word = letter*n+word
                        updated[new_word] = updated.get(new_word, mp.mpf(0))+value*time**n/math.factorial(n)
                coefficients = updated
        errors = [mp.mpf(0)]*(degree+1)
        for word, value in coefficients.items():
            errors[len(word)] = max(errors[len(word)], abs(value-mp.mpf(1)/math.factorial(len(word))))
        return {str(n): mp.nstr(errors[n], 12) for n in range(1, degree+1)}
