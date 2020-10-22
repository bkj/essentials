%module test

%{
/* Includes the header in the wrapper code */
#define SWIG_FILE_WITH_INIT
#include "test.cuh"
%}

%include "test.cuh"

%template(do_testI) do_test<int>;
%template(do_testF) do_test<float>;