#include <CL/sycl.hpp>
// #include <tick_count.h>
#include <array>
#include <iostream>
#include <limits>
#include <stdio.h>

#include <stdint.h>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

 //////////////////////////////////////////////////////////////
// Filter Code Definitions
//////////////////////////////////////////////////////////////
// maximum number of inputs that can be handled
// in one function call
#define MAX_INPUT_LEN 80
// maximum length of filter than can be handled
#define MAX_FLT_LEN 3
// buffer to hold all of the input samples
#define BUFFER_LEN (MAX_FLT_LEN - 1 + MAX_INPUT_LEN)

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

// FIR init
void firFloatInit(double *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = 0;
  a[0] = 1;
  // std::memset(insamp, 0, sizeof(insamp));
}

// ACC init
void AccFloatInit(double *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = 0;
}

// the FIR filter function
void firFloat(double * coeffs, double * input, double * output,
  int length, int filterLength, queue &q) {
  double* acc = malloc_shared<double>(sizeof(double)*filterLength, q); // accumulator for MACs allocated on device memory

  //Init acc with 0
  AccFloatInit(acc, filterLength);
  
  q.submit([&](handler &h){

    h.parallel_for(range<1>(length), [=, &q](auto i){
      double total = 0;
      //Multiply
      handler h;
      h.parallel_for(range<1>(filterLength), [=](auto j){
        int x = i[0] - j[0]; // position in input
        
        if (x >= 0)
        {
          acc[j] += coeffs[j] * input[x];
        }
      });

      //Accumulate
      for (int k = 0; k < filterLength; k++) {
        total += acc[k];
      }
      output[i] = total;
      AccFloatInit(acc, filterLength);
    });
  });
  q.wait();

};


//////////////////////////////////////////////////////////////
// Test program
//////////////////////////////////////////////////////////////
// bandpass filter centred around 1000 Hz
// sampling rate = 8000 Hz
#define FILTER_LEN 3
double coeffs[FILTER_LEN] = {-0.0448093,0.0322875,0.0181163};

void intToFloat(int16_t * input, double * output, int length) {
  int i;
  for (i = 0; i < length; i++) {
    output[i] = (double) input[i];
  }
}
void floatToInt(double * input, int16_t * output, int length) {
  int i;
  for (i = 0; i < length; i++) {
    // add rounding constant
    input[i] += 0.5;
    // bound the values to 16 bits
    if (input[i] > 32767.0) {
      input[i] = 32767.0;
    } else if (input[i] < -32768.0) {
      input[i] = -32768.0;
    }
    // convert
    output[i] = (int16_t) input[i];
  }
}
// number of samples to read per loop
#define SAMPLES 80
int main(void) {

  // The default device selector will select the most performant device.
  default_selector d_selector;
  queue q(d_selector, exception_handler);
  // Print out the device information used for the kernel code.
  std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>() << "\n";

  try{

    size_t size;
    double *input = malloc_shared<double>(SAMPLES, q);
    double *output = malloc_shared<double>(SAMPLES, q);
    double *floatInput = malloc_shared<double>(SAMPLES, q);
    double *floatOutput = malloc_shared<double>(SAMPLES, q);
    
    // initialize the filter
    firFloatInit(input, SAMPLES);
    // process all of the samples
    do {
      size = 0;
      firFloat(coeffs, input, output, SAMPLES, FILTER_LEN, q);
    } while (size != 0);
    
    for(int i = 0; i < SAMPLES; i++){
      std::cout << output[i] << ", ";
    }
    std::cout << "\n";
  }
  catch (exception const &e)
  {
    std::cout << "An exception is caught while filtering.\n";
    std::terminate();
  }
  std::cout << "filtering successfull.\n";
  return 0;
}