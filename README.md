## How to use
Clone the repo and update submodules:
```
git clone git@github.com:pedrokwerle/axonc.git
cd axonc
git submodule update --init --recursive
```

Make onnx2c:
```
make onnx2c
```

Try convert example model:
```
make example MODEL_NAME=squeezenet1_0_Opset16_fp16_unpadded
```

Try with other models:
```
make MODEL=<model_name>
```

## Notes
For now only supports FP16 SAURIA, to convert onnx model to fp16 use script in scripts directory.
SAURIA does not support padded convolutions, to isolate padded convolutions to seaprate padding layers use legalization script. 
Some models don't have named nodes, this is a problem for this pipeline, so use naming script to name all nodes.
