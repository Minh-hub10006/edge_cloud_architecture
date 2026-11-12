# SNPE persistent inference daemon (QAIRT 2.43 C API)
# Build on edge-246: scripts/build_snpe_daemon_edge.sh

QAIRT ?= $(HOME)/sigm/qairt/2.43.0.260128
ABI   ?= aarch64-oe-linux-gcc11.2
CXX   ?= g++

INCLUDES := -I$(QAIRT)/include/zdl -I$(QAIRT)/include/SNPE
LDFLAGS  := -L$(QAIRT)/lib/$(ABI) -Wl,-rpath,$(QAIRT)/lib/$(ABI)
LIBS     := -lSNPE -lpthread

CXXFLAGS := -std=c++17 -O2 -Wall -Wextra $(INCLUDES)

.PHONY: all clean

all: snpe-daemon

snpe-daemon: snpe_daemon.cpp
	$(CXX) $(CXXFLAGS) snpe_daemon.cpp $(LDFLAGS) $(LIBS) -o $@

clean:
	rm -f snpe-daemon
