import onnx
model = onnx.load("/home/sana/Yan/crowd/crowd density.v1-v1.yolo26/ameba-ai-offline-toolkit/Docker/Linux/acuity_examples_c901149/best.onnx")

print("Looking for Conv nodes right before the final concatenation/DFL...")

conv_outputs = []
for node in model.graph.node:
    if node.op_type == "Conv":
        conv_outputs.append((node.name, node.output[0]))

print("Last 20 Conv outputs:")
for name, out in conv_outputs[-20:]:
    print(f"Conv Name: {name}, Output: {out}")

# Or just print the last 30 nodes overall
print("\nLast 30 nodes overall:")
for node in model.graph.node[-30:]:
    print(f"Op: {node.op_type}, Name: {node.name}, Outputs: {node.output}")
