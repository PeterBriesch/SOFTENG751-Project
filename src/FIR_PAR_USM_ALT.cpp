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
// maximum length of filter than can be handled
// buffer to hold all of the input samples
constexpr int FILTER_LEN = 256;
constexpr int SAMPLES = 2560000;
constexpr int N = SAMPLES; //global size
constexpr int P = FILTER_LEN;

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
  // for (size_t i = 0; i < size; i++) a[i] = (rand() % 1);
  for (size_t i = 0; i < size; i++) a[i] = 0;

  a[1] = 1;
  // std::memset(insamp, 0, sizeof(insamp));
}

// ACC init
void AccFloatInit(double *a, size_t size) {
  for (size_t i = 0; i < size; i++) a[i] = 0;
}

// the FIR filter function
void firFloat(double * coeffs, double * input, double * output,
  int length, int filterLength, double* acc, int* j, queue &q) {


  for (int j = 0 ; j < length ; j++){
      output[j] = 0;
      // calculate output n 
      q.submit([&](auto &h){
        //int acc = 0;
        h.parallel_for(range(filterLength), [=](auto i){
          if(j-i[0] >= 0)
          {
            output[j] += coeffs[i] * input[j-i[0]];
          }
        });
        //output[j] = acc;
      });
      q.wait();
  };

};


//////////////////////////////////////////////////////////////
// Test program
//////////////////////////////////////////////////////////////
// bandpass filter centred around 1000 Hz
// sampling rate = 8000 Hz

double coeffs[FILTER_LEN]; //= {-0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093,-0.0448093, -0.0448093,-0.0448093,-0.0448093,-0.0448093};


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

int main(void) {
  srand(1);
  for(int i = 0; i < FILTER_LEN; i++) coeffs[i] = -0.0448093;

  // The default device selector will select the most performant device.
  default_selector d_selector;
  queue q(d_selector, exception_handler);
  // Print out the device information used for the kernel code.
  std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>() << "\n";

  try{

    size_t size;
    //alocate memory on device for use in parallel_for
    double* acc = static_cast<double *>(malloc_shared(sizeof(double), q)); // accumulator for MACs allocated on device memory
    int* j = static_cast<int *>(malloc_shared(sizeof(int), q));
    double* input_device = static_cast<double *>(malloc_shared(sizeof(double)*SAMPLES, q));
    double* output_device = static_cast<double *>(malloc_shared(sizeof(double)*SAMPLES, q));
    double* coeffs_device = static_cast<double *>(malloc_shared(sizeof(double)*FILTER_LEN, q));
    q.memcpy(coeffs_device, coeffs, sizeof(double)*FILTER_LEN).wait();
    
    // initialize the filter
    firFloatInit(input_device, SAMPLES);
    
    // process all of the samples
    do {

      size = 0;
      auto start = high_resolution_clock::now();
      firFloat(coeffs_device, input_device, output_device, SAMPLES, FILTER_LEN, acc, j, q);
      auto stop = high_resolution_clock::now();

      auto duration = duration_cast<microseconds>(stop-start);
      std::cout << "Duration of Filtering = " << duration.count() << " microseconds\n\n";
    } while (size != 0);
    
    for(int i = 0; i < SAMPLES; i++){
      std::cout << output_device[i] << ", ";
    }
    std::cout << "\n";

    free(acc, q);
    free(j, q);
    free(output_device, q);
    free(input_device, q);
    free(coeffs_device, q);
  }
  catch (exception const &e)
  {
    std::cout << "An exception is caught while filtering.\n";
    std::terminate();
  }
  std::cout << "filtering successfull.\n";
  return 0;
}