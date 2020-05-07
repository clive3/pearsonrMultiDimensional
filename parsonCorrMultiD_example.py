import numpy as np


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


class pearsonrMultiD(object):


    def run(self):

        ## in this example a 1-dimensional array of order 7 is used
        common_dimension: int=7
        array_multid = np.random.randint(100, size=(common_dimension, 5, 2))
        array_1d = np.random.randint(100, size=common_dimension)

        correlation_array = pearsonCorrMultiD(array_1d, array_multid)

        print(array_1d)
        print(array_multid)
        print(correlation_array)


if __name__ == '__main__':

    # make an instance of the class and implement the run function
    obj = pearsonrMultiD()
    obj.run()
