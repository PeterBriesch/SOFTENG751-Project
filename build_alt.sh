#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
rm build_alt.sh.*
rm run_alt.sh.*
make build_FIR_par_usm_alt
