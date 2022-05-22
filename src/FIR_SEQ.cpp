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
constexpr int FILTER_LEN = 256;
constexpr int SAMPLES = 2560000;

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

// the FIR filter function
void firFloat(double * coeffs, double * input, double * output,
  int length, int filterLength) {
  double acc; // accumulator for MACs
  int n, k, j;

  // apply the filter to each input sample
  for (n = 0; n < length; n++) {
    // calculate output n
    acc = 0;
    for (k = 0; k < filterLength; k++) {
      j = n - k; // position in input
      
      if (j >= 0)
      {
        acc += coeffs[k] * input[j];
      }
    }
    output[n] = acc;
    
  }
  // // shift input samples back in time for next time
  // std::memmove( & insamp[0], & insamp[length],
  //   (filterLength - 1) * sizeof(double));
}


//////////////////////////////////////////////////////////////
// Test program
//////////////////////////////////////////////////////////////
// bandpass filter centred around 1000 Hz
// sampling rate = 8000 Hz
double coeffs[FILTER_LEN];

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

    // array to hold input samples
    // double *insamp = malloc_shared<double>(BUFFER_LEN, q);
    size_t size;
    double *input = static_cast<double *>(malloc(SAMPLES*sizeof(double)));
    double *output = static_cast<double *>(malloc(SAMPLES*sizeof(double)));
    // FILE * in_fid;
    // FILE * out_fid;
    // open the input waveform file
    // in_fid = fopen("input.pcm", "rb");
    // if (in_fid == 0) {
    //   printf("couldn't open input.pcm");
    //   return -1;
    // }
    // // open the output waveform file
    // out_fid = fopen("outputFloat.pcm", "wb");
    // if (out_fid == 0) {
    //   printf("couldn't open outputFloat.pcm");
    //   return -1;
    // }
    // initialize the filter
    firFloatInit(input, SAMPLES);
    // process all of the samples
    do {
      // read samples from file
      // size = fread(input, sizeof(int16_t), SAMPLES, in_fid);
      size = 0;
      // convert to doubles
      // intToFloat(input, floatInput, size);
      // perform the filtering
      auto start = high_resolution_clock::now();
      firFloat(coeffs, input, output, SAMPLES, FILTER_LEN);
      auto stop = high_resolution_clock::now();

      auto duration = duration_cast<microseconds>(stop-start);
      std::cout << "Duration of Filtering = " << duration.count() << " microseconds\n\n";
      // convert to ints
      // floatToInt(floatOutput, output, size);
      // write samples to file
      // fwrite(output, sizeof(int16_t), size, out_fid);
    } while (size != 0);
    for(int i = 0; i < SAMPLES; i++){
      std::cout << output[i] << ", ";
    }
    std::cout << "\n";
    // fclose(in_fid);
    // fclose(out_fid);
    free(input);
    free(output);
    
  }
  catch (exception const &e)
  {
    std::cout << "An exception is caught while filtering.\n";
    std::terminate();
  }
  std::cout << "filtering successfull.\n";
  return 0;
}