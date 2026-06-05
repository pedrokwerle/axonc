.PHONY: all example clean clean-all onnx2c

ONNX2C      := onnx2c/build/onnx2c
HW_TARGET   := FP16_8x16
EXAMPLE_DIR := example_models
GEN_DIR     := generated
DEFAULT_EX  := squeezenet1_0_Opset16_fp16_unpadded

# ---------- all: requires MODEL=path/to/model.onnx ----------
ifdef MODEL
  _NAME := $(basename $(notdir $(MODEL)))
  _CONF := $(GEN_DIR)/$(_NAME).conf
  _MANI := $(GEN_DIR)/$(_NAME)/conv_manifest.json
  _CSRC := $(GEN_DIR)/$(_NAME)/$(_NAME).c

all: $(ONNX2C) $(_CSRC)

$(_MANI): $(MODEL) | $(GEN_DIR)/$(_NAME)/
	cd sauria_gen && python sauria_gen.py \
		-o ../$(GEN_DIR)/$(_NAME) \
		-hw $(HW_TARGET) \
		--sauria-path sauria_ucd/Python/src/ \
		../$(MODEL) \
		> ../$(GEN_DIR)/$(_NAME)/$(_NAME).conf

$(_CSRC): $(MODEL) $(_MANI) | $(GEN_DIR)/$(_NAME)/
	$(ONNX2C) $(MODEL) \
		-P -t sauria \
		-m $(GEN_DIR)/$(_NAME)/conv_manifest.json \
		> $(_CSRC)

else
all:
	@echo "Usage: make MODEL=path/to/model.onnx"
	@echo "       make example [EX=model_name]"
endif

# ---------- example ----------
EX ?= $(DEFAULT_EX)

example: $(ONNX2C)
	$(MAKE) MODEL=$(EXAMPLE_DIR)/$(EX).onnx $(GEN_DIR)/$(EX)/$(EX).c

# ---------- directory rule ----------
$(GEN_DIR)/%/:
	mkdir -p $@

# ---------- onnx2c build ----------
onnx2c: $(ONNX2C)

$(ONNX2C): onnx2c/CMakeLists.txt
	cd onnx2c && git submodule update --init
	mkdir -p onnx2c/build
	cd onnx2c/build && cmake -DCMAKE_BUILD_TYPE=Release .. && make onnx2c

onnx2c/CMakeLists.txt:
	git submodule update --init

# ---------- clean ----------
clean:
	rm -rf $(GEN_DIR)

clean-all: clean
	rm -rf onnx2c/build
