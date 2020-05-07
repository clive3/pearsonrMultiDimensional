def pearsonCorrMultiD(array_1D, array_multiD):

    ## normalise the two arrays
    norm_array_multiD = array_multiD - np.mean(array_multiD, axis=0)
    norm_array_1D = array_1D - np.mean(array_1D)

    ## in order to perform the array multiplication and be able to sum over axis=0
    ## the multi-dimensional array needs to be transposed so that the common dimension
    ## is the last dimension not the first
    trans_norm_array_multiD = np.transpose(norm_array_multiD)

    ## this is pearson correlation performed on arrays
    ##
    ## note the numerator needs to transpose back before summing over axis=0
    numerator = np.sum(np.transpose((trans_norm_array_multiD * norm_array_1D)), axis=0)
    denominator = np.sqrt(np.sum((norm_array_multiD ** 2), axis=0)) * np.sqrt(np.sum((norm_array_1D ** 2)))
    pearsonr_multiD_np = numerator / denominator

    return pearsonr_multiD_np
