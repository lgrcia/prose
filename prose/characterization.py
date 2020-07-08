class Characterize:

    def __init__(self):
        """
        Base class for characterization routines"""
        pass

    def run(self, image):
        """
        Run characterisation by returning a dictionary like
        {
            "property_0": value_0,
            "property_1": value_1,
            "property_2": value_2,
        }

        to be added to reduced_image FITS

        Parameters
        ----------
        image : np.ndarray
            Image to be characterized

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("This method should be overwritten")