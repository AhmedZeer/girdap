ifndef RISCV
$(error RISCV is unset)
else
$(info Running with RISCV=$(RISCV))
endif

default: libtoyrocc.so

libtoyrocc.so: libtoyrocc.cc
	g++ -L $(RISCV)/lib -Wl,-rpath,$(RISCV)/lib -shared -o $@ -std=c++17 -I $(RISCV)/include -fPIC -lriscv -O3 $^

.PHONY: install
install: libtoyrocc.so
	cp libtoyrocc.so $(RISCV)/lib

clean:
	rm -rf *.o *.so
