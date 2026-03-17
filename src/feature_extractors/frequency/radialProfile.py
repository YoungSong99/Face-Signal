# Based on:
# https://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles
#
# Modifications: Replaced integer-radius binning with normalized radial bins (r_norm)
import numpy as np

def azimuthalAverage(image, center=None, num_bins=32):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])
    r_norm = r / (r.max() + 1e-8)
        
    bin_idx = (r_norm * num_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    sums = np.bincount(bin_idx.ravel(), weights=image.ravel(), minlength=num_bins)
    counts = np.bincount(bin_idx.ravel(), minlength=num_bins)

    radial = sums / (counts + 1e-8)

    return radial
