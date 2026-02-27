RISCV_PFX     ?= riscv64-unknown-elf-
RISCV_CC      := $(RISCV_PFX)gcc
RISCV_OBJDUMP := $(RISCV_PFX)objdump

# Directories
SRC_DIR   := src
BUILD_DIR := build

# Find all .c files in src/
SRCS      := $(wildcard $(SRC_DIR)/*.c)

# Create a list of target ELFs and DUMPs based on the source filenames
# src/foo.c -> build/foo.riscv
ELFS      := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.riscv)
DUMPS     := $(ELFS:.riscv=.dump)

CFLAGS        ?= -O2 -static -specs=htif_nano.specs -Iinclude
OBJDUMP_FLAGS ?= --disassemble-all

.PHONY: all clean

# This will build EVERY .riscv and EVERY .dump for every .c found
all: $(ELFS) $(DUMPS)

# Pattern rule: How to turn a src/%.c into a build/%.riscv
$(BUILD_DIR)/%.riscv: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(RISCV_CC) $(CFLAGS) -o $@ $<

# Pattern rule: How to turn a .riscv into a .dump
$(BUILD_DIR)/%.dump: $(BUILD_DIR)/%.riscv
	$(RISCV_OBJDUMP) $(OBJDUMP_FLAGS) $< > $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)