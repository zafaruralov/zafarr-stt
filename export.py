# Turn the NeMo model (needs PyTorch, ~2 GB) into ONNX (needs only onnxruntime, small).
import nemo.collections.asr as nemo_asr
m = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_uz_fastconformer_hybrid_large_pc", map_location="cpu")
m.set_export_config({"decoder_type": "ctc"})   # hybrid model: export the simple CTC head
m.export("model.onnx")
with open("vocab.txt", "w") as f:
    for i, t in enumerate(m.ctc_decoder.vocabulary if hasattr(m, "ctc_decoder") else m.decoder.vocabulary):
        f.write(f"{t} {i}\n")
    f.write(f"<blk> {i+1}\n")
print(m.cfg.preprocessor)

# Then (in another venv with onnxruntime) shrink it to int8, ~175 MB:
# from onnxruntime.quantization import quantize_dynamic, QuantType
# quantize_dynamic("model.onnx", "model.int8.onnx", weight_type=QuantType.QInt8,
#                  op_types_to_quantize=["MatMul"], per_channel=True)
