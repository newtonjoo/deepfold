#which compiler
intel=intel11.1
#CC=/usr/local/${intel}/bin/intel64/icc
CC=icc
CPP=icc

GCC=-g -ipo

#ICC=-O3 -fp-model fast -xHost -ipo -no-prec-div

# where libraries are located
#LIBDIR=/usr/local/${openmpi}/lib
#LIBDIR=/usr/local/${intel}/lib/intel64
LIBDIR=
#LIBDIR=/usr/local/mpich-gnu/lib
#where  are the include files
#INCLUDE=/usr/local/${intel}/include
#INCLUDE=/usr/local/mpich-intel/include
INCLUDE=src_hhm/cs
IN_HHM=src_hhm

# Options for compilation

CFLAGS=-g -ipo -fPIC -shared -L $(LIBDIR) $(LFLAGS) -I$(INCLUDE) -I$(IN_HHM) -no-multibyte-chars
LFLAGS=

.SUFFIXES:	.cpp 

CS_DIR = src_hhm/cs
CS_SOURCES_WITH_APPS = $(filter-out $(wildcard $(CS_DIR)/*test*.cc), $(wildcard $(CS_DIR)/*.cc))
CS_SOURCES = $(filter-out $(wildcard $(CS_DIR)/*app.cc), $(CS_SOURCES_WITH_APPS))
CS_OBJECTS = $(patsubst %.cc, %.o, $(CS_SOURCES))

.cpp.o:
	$(CC) $(CFLAGS) -I$(INCLUDE) -c $<

TARGET = pCRF_pair_align.so

ALL: $(TARGET)

######################################################################
######################################################################

pCRF_pair_align.so : pCRF_pair_align.o $(CS_OBJECTS)
	$(CC) $(LFLAGS) -shared -fPIC -Wno-format-security -Wno-vla -o pCRF_pair_align.so pCRF_pair_align.o $(CS_OBJECTS) -L $(LIBDIR) $(LFLAGS) -I. -I$(INCLUDE) -I$(IN_HHM)  -lintlc -no-multibyte-chars

pCRF_pair_align.o : pCRF_pair_align.cpp
	 $(CC) -DMAIN -g -I. -I$(INCLUDE) -fPIC -c pCRF_pair_align.cpp -o  pCRF_pair_align.o  -lintlc -no-multibyte-chars -Wwrite-strings

$(CS_OBJECTS): $(CS_DIR)/%.o: $(CS_DIR)/%.cc
	$(CPP) -g -DMAIN -fPIC $(OBJECT_LIBS) -c $< -o $@ -Wno-deprecated -lintlc -no-multibyte-chars -Write-strings

clean:
	rm -f $(TARGET) *.o $(CS_OBJECTS)

	
