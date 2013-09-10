// Disable atomic operations and big integer support.
// It doesn't work with nvcc.
//
// This file is pre-included before any other header.

#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
