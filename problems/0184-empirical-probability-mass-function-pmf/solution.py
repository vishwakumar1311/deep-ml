from tinygrad.tensor import Tensor

def empirical_pmf(samples: Tensor) -> list:
    """
    Given a Tensor of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    sample_list = samples.flatten().tolist()
    sorted(sample_list)
    total_count = len(sample_list)
    counts={}
    for val in sample_list:
        counts[val] = counts.get(val,0)+1
    
    result = [(val,(count/total_count)) for val,count in sorted(counts.items())]
    return (result)
    pass