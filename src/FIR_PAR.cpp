#include <CL/sycl.hpp>
#include <chrono>
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
using namespace std::chrono;

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
  for (size_t i = 0; i < size; i++) a[i] = (rand() % 1);
  // std::memset(insamp, 0, sizeof(insamp));
}

// ACC init
void AccFloatInit(double *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = 0;
}

// the FIR filter function
void firFloat(double * coeffs, double * input, double * output,
  int length, int filterLength, double* acc, int* j, queue &q) {

  q.parallel_for(length, [=](auto i){
    // calculate output n
    *acc = 0;
    #pragma unroll
    for (int k = 0; k < filterLength; k++) {
      *j = i[0] - k; // position in input
      
      if (*j >= 0)
      {
        *acc += coeffs[k] * input[*j];
      }
    }
    output[i] = *acc;

  }).wait();

};


//////////////////////////////////////////////////////////////
// Test program
//////////////////////////////////////////////////////////////
// bandpass filter centred around 1000 Hz
// sampling rate = 8000 Hz
#define FILTER_LEN 1
double coeffs[FILTER_LEN] = {-0.0448093};

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
#define SAMPLES 999999999
int main(void) {
  srand(1);

  // The default device selector will select the most performant device.
  default_selector d_selector;
  queue q(d_selector, exception_handler);
  // Print out the device information used for the kernel code.
  std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>() << "\n";

  try{

    size_t size;
    double *input = static_cast<double *>(malloc(SAMPLES * sizeof(double)));
    double *output = static_cast<double *>(malloc(SAMPLES * sizeof(double)));

    //alocate memory on device for use in parallel_for
    double* acc = malloc_device<double>(sizeof(double), q); // accumulator for MACs allocated on device memory
    int* j = malloc_device<int>(sizeof(int), q);
    double* input_device = malloc_device<double>(sizeof(double) *SAMPLES, q);
    double* output_device = malloc_device<double>(sizeof(double) *SAMPLES, q);
    double* coeffs_device = malloc_device<double>(sizeof(double)*SAMPLES, q);
    q.memcpy(coeffs_device, coeffs, sizeof(double)*FILTER_LEN).wait();
    
    // initialize the filter
    firFloatInit(input, SAMPLES);
    
    // process all of the samples
    do {

      q.memcpy(input_device, input, sizeof(double)*SAMPLES).wait();
      std::cout << "memcpy 1 complete\n";
      size = 0;
      auto start = high_resolution_clock::now();
      firFloat(coeffs, input, output_device, SAMPLES, FILTER_LEN, acc, j, q);
      auto stop = high_resolution_clock::now();

      q.memcpy(output, output_device, sizeof(double) * SAMPLES).wait();

      auto duration = duration_cast<microseconds>(stop-start);
      std::cout << "Duration of Filtering = " << duration.count() << " microseconds\n\n";
    } while (size != 0);
    
    // for(int i = 0; i < SAMPLES; i++){
    //   std::cout << output[i] << ", ";
    // }
    // std::cout << "\n";

    free(input);
    free(output);
    free(acc, q);
    free(j, q);
    free(output_device, q);
  }
  catch (exception const &e)
  {
    std::cout << "An exception is caught while filtering.\n";
    std::terminate();
  }
  std::cout << "filtering successfull.\n";
  return 0;
}