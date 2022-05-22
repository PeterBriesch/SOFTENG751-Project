#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
rm build.sh.*
rm run.sh.*
make build_FIR_par_usm
