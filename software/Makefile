BUILD_MODE ?= baremetal

RISCV_ELF_PFX ?= riscv64-unknown-elf-
RISCV_LINUX_GCC ?= $(shell command -v riscv64-unknown-linux-gnu-gcc 2>/dev/null)
RISCV_LINUX_OBJDUMP ?= $(shell command -v riscv64-unknown-linux-gnu-objdump 2>/dev/null)

ifeq ($(BUILD_MODE),baremetal)
RISCV_CC      := $(RISCV_ELF_PFX)gcc
RISCV_OBJDUMP := $(RISCV_ELF_PFX)objdump
CFLAGS        ?= -O2 -static -specs=htif_nano.specs -Iinclude
else ifeq ($(BUILD_MODE),linux)
ifeq ($(RISCV_LINUX_GCC),)
RISCV_CC      := riscv64-linux-gnu-gcc
RISCV_OBJDUMP := riscv64-linux-gnu-objdump
else
RISCV_CC      := riscv64-unknown-linux-gnu-gcc
ifeq ($(RISCV_LINUX_OBJDUMP),)
RISCV_OBJDUMP := riscv64-linux-gnu-objdump
else
RISCV_OBJDUMP := riscv64-unknown-linux-gnu-objdump
endif
endif
CFLAGS        ?= -O2 -static -Iinclude
else
$(error Unsupported BUILD_MODE '$(BUILD_MODE)'; use 'baremetal' or 'linux')
endif

# Directories
SRC_DIR          := src
COMMON_DIR       := common
BUILD_DIR        := build
COMMON_BUILD_DIR := $(BUILD_DIR)/common

# Find all .c files in src/
SRCS        := $(wildcard $(SRC_DIR)/*.c)
COMMON_SRCS := $(wildcard $(COMMON_DIR)/*.c)

# Create a list of target ELFs and DUMPs based on the source filenames
# src/foo.c -> build/foo.riscv
ELFS        := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.riscv)
DUMPS       := $(ELFS:.riscv=.dump)
COMMON_OBJS := $(COMMON_SRCS:$(COMMON_DIR)/%.c=$(COMMON_BUILD_DIR)/%.o)
DEPS        := $(ELFS:.riscv=.d) $(COMMON_OBJS:.o=.d)

OBJDUMP_FLAGS ?= --disassemble-all

.PHONY: all clean

# This will build EVERY .riscv and EVERY .dump for every .c found
all: $(ELFS) $(DUMPS)

# Pattern rule: How to turn a src/%.c into a build/%.riscv
$(BUILD_DIR)/%.riscv: $(SRC_DIR)/%.c $(COMMON_OBJS) | $(BUILD_DIR) $(COMMON_BUILD_DIR)
	$(RISCV_CC) $(CFLAGS) -MMD -MP -MF $(@:.riscv=.d) -o $@ $< $(COMMON_OBJS) -lm

$(COMMON_BUILD_DIR)/%.o: $(COMMON_DIR)/%.c | $(COMMON_BUILD_DIR)
	$(RISCV_CC) $(CFLAGS) -MMD -MP -MF $(@:.o=.d) -c -o $@ $<

# Pattern rule: How to turn a .riscv into a .dump
$(BUILD_DIR)/%.dump: $(BUILD_DIR)/%.riscv
	$(RISCV_OBJDUMP) $(OBJDUMP_FLAGS) $< > $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $@

$(COMMON_BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
	rm -f *.riscv
	rm -f libgirdap/*.so

-include $(DEPS)
