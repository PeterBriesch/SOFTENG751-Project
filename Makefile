CXX = dpcpp
CXXFLAGS = -O2 -g -std=c++17

SEQ_EXE_NAME = seq_fir
SEQ_SOURCES = src/FIR_SEQ.cpp

PAR_EXE_NAME = par_fir
PAR_SOURCES = src/FIR_PAR.cpp

all: build_FIR_seq

build_FIR_seq:
	$(CXX) $(CXXFLAGS) -o $(SEQ_EXE_NAME) $(SEQ_SOURCES)

build_FIR_par:
	$(CXX) $(CXXFLAGS) -o $(PAR_EXE_NAME) $(PAR_SOURCES)

run: 
	./$(SEQ_EXE_NAME)

run_par: 
	./$(PAR_EXE_NAME)

clean: 
	rm -f $(SEQ_EXE_NAME) $(PAR_EXE_NAME)
