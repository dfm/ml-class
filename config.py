#
# If you want to compile with a C implementation of the EM algorithm, set
# LAPACK to the path to your clapack.h header file.  For example, on my Mac,
# the following path works:
#   LAPACK='/System/Library/Frameworks/vecLib.framework/Versions/A/Headers'
# The code doesn't really work with all versions of CLAPACK so I'm not too sure
# whether this will work for you or not...
#

LAPACK=None

