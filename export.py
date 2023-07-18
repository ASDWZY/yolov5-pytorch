import torch


def export_onnx(model, onnx_file, simplify=False):
    import onnx
    x = torch.rand((1, 3, 640, 640))
    torch.onnx.export(
        model, x,
        onnx_file,
        verbose=False,
        opset_version=12,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["preds"],
        dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'}, "preds": {0: 'batch', 1: 'anchors'}}
    )
    if simplify:
        import onnxsim
        model_onnx = onnx.load(onnx_file)
        model_onnx, check = onnxsim.simplify(model_onnx)
        if not check:
            raise Exception("onnx model simplify failed")
        onnx.save(model_onnx, onnx_file)

# def export_tensorrt(model, tensorrt_):



if __name__ == '__main__':
    # export_onnx("yolov5_cls_40.pth", "cls.onnx", True)
    export_onnx("best.pt", "fire0704.onnx", False)
