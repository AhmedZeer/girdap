RISCV_PFX ?= riscv64-unknown-elf-
RISCV_CC := $(RISCV_PFX)gcc
RISCV_OBJDUMP := $(RISCV_PFX)objdump

SRC ?= basic.c
TARGET ?= $(basename $(notdir $(SRC)))

CFLAGS ?= -O2 -static -specs=htif_nano.specs -Iinclude
OBJDUMP_FLAGS ?= --disassemble-all

ELF := $(TARGET).riscv
DUMP := $(ELF).dump
JUNK := $(ELF) $(DUMP)

.PHONY: all clean run

all: $(ELF) $(DUMP)

$(ELF): $(SRC)
	$(RISCV_CC) $(CFLAGS) -o $@ $<

$(DUMP): $(ELF)
	$(RISCV_OBJDUMP) $(OBJDUMP_FLAGS) $< > $@

# Example:
#   make run RUNNER="spike --extension=toyrocc "
run: $(ELF)
	$(RUNNER)$<

clean:
	rm -f $(JUNK)
