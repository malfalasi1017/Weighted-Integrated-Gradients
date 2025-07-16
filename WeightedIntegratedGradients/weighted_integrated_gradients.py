import numpy as np

def weighted_integrated_gradients(
    inp, 
    target_label_index,
    predictions_and_gradients,
    baseline,
    weighting_function=None,
    steps=50):

    if baseline is None:
        baseline = 0*inp
    assert(baseline.shape == inp.shape)

    
    scaled_inputs = [baseline + (float(i)/steps)*(inp-baseline) for i in range(0, steps+1)] 
    predictions, grads = predictions_and_gradients(scaled_inputs, target_label_index) 

    grads = (grads[:-1] + grads[1:]) / 2.0 

    if weighting_function is not None: 
        weights = np.array([weighting_function(k/steps) for k in range(1, steps+1)])
        
        Z = np.sum(weights)
        weights = weights / Z
        
        weighted_grads = np.zeros_like(grads[0])
        for i in range(len(grads)):
            weighted_grads += weights[i] * grads[i]
        
        avg_grads = weighted_grads
    else:
        avg_grads = np.average(grads, axis=0)

    weighted_integrated_gradients = (inp-baseline)*avg_grads  
    
    return weighted_integrated_gradients, predictions

def random_baseline_weighted_integrated_gradients(
    inp, 
    target_label_index,
    predictions_and_gradients,
    steps=50,
    num_random_trials=10,
    weighting_function=None):
    all_intgrads = []
    for i in range(num_random_trials):
        intgrads, prediction_trend = weighted_integrated_gradients(
            inp, 
            target_label_index=target_label_index,
            predictions_and_gradients=predictions_and_gradients,
            baseline=255.0*np.random.random(inp.shape), 
            steps=steps,
            weighting_function=weighting_function)
        all_intgrads.append(intgrads)
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads

def sqrt_weighting_function(alpha):
    """
    A weighting function that returns the square root of alpha.
    """
    return np.sqrt(alpha)

def reciprocal_weighting_function(alpha):
    """
    A reciprocal weighting function that returns the reciprocal of alpha.
    """
    return 1.0 / alpha if alpha > 0 else 0.0

def linear_late_weighting_function(alpha):
    """Linear weighting emphasizing near the input"""
    return alpha

def linear_early_weighting_function(alpha):
    """Linear weighting emphasizing near the baseline"""
    return 1 - alpha

def quadratic_early_weighting_function(alpha):
    """Quadratic weighting emphasizing near the baseline"""
    return (1 - alpha) ** 2

def quadratic_late_weighting_function(alpha):
    """Quadratic weighting emphasizing near the input"""
    return alpha ** 2

def logarithmic_weighting_function(alpha, c=1.0):
    """Logarithmic weighting with sharp rise near input"""
    return np.log(1 + c * alpha)

def exponential_weighting_function(alpha, beta=1.0):
    """Exponential weighting focusing near the input"""
    return np.exp(beta * (alpha - 1))

def beta22_weighting_function(alpha):
    """Beta(2,2) weighting emphasizing middle of the path"""
    return 6 * alpha * (1 - alpha)
