CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

SEQ_EXE_NAME = seq_fir
SEQ_SOURCES = src/FIR_SEQ.cpp

PAR_EXE_NAME = par_fir
PAR_SOURCES = src/FIR_PAR.cpp

PAR_EXE_USM_NAME = par_fir_usm
PAR_USM_SOURCES = src/FIR_PAR_USM.cpp

PAR_EXE_USM_ALT_NAME = par_fir_usm_alt
PAR_USM_ALT_SOURCES = src/FIR_PAR_USM_ALT.cpp

all: build_FIR_seq build_FIR_par_usm build_FIR_par_usm_alt

build_FIR_seq:
	$(CXX) $(CXXFLAGS) -o $(SEQ_EXE_NAME) $(SEQ_SOURCES)

build_FIR_par:
	$(CXX) $(CXXFLAGS) -o $(PAR_EXE_NAME) $(PAR_SOURCES)
	
build_FIR_par_usm:
	$(CXX) $(CXXFLAGS) -o $(PAR_EXE_USM_NAME) $(PAR_USM_SOURCES)

build_FIR_par_usm_alt:
	$(CXX) $(CXXFLAGS) -o $(PAR_EXE_USM_ALT_NAME) $(PAR_USM_ALT_SOURCES)

run: run_seq run_par_usm run_par_usm_alt

run_seq:
	./$(SEQ_EXE_NAME)

run_par: 
	./$(PAR_EXE_NAME)

run_par_usm:
	./$(PAR_EXE_USM_NAME)

run_par_usm_alt:
	./$(PAR_EXE_USM_ALT_NAME)
clean: 
	rm -f $(SEQ_EXE_NAME) $(PAR_EXE_NAME) $(PAR_EXE_USM_NAME) $(PAR_EXE_USM_ALT_NAME) *.sh.*
