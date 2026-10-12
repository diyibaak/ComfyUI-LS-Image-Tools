import numpy as np
from PIL import Image, ImageOps


STANDARD_IMAGE_SIZES = {
    "1K": (
        ("1:1", 1 / 1, 1024, 1024),
        ("4:3", 4 / 3, 1184, 896),
        ("3:4", 3 / 4, 896, 1184),
        ("5:4", 5 / 4, 1152, 928),
        ("4:5", 4 / 5, 928, 1152),
        ("3:2", 3 / 2, 1248, 832),
        ("2:3", 2 / 3, 832, 1248),
        ("16:9", 16 / 9, 1376, 768),
        ("9:16", 9 / 16, 768, 1376),
        ("21:9", 21 / 9, 1568, 672),
        ("9:21", 9 / 21, 672, 1568),
        ("2:1", 2 / 1, 1440, 736),
        ("1:2", 1 / 2, 736, 1440),
    ),
    "1.5K": (
        ("1:1", 1 / 1, 1536, 1536),
        ("4:3", 4 / 3, 1776, 1328),
        ("3:4", 3 / 4, 1328, 1776),
        ("5:4", 5 / 4, 1728, 1376),
        ("4:5", 4 / 5, 1376, 1728),
        ("3:2", 3 / 2, 1872, 1248),
        ("2:3", 2 / 3, 1248, 1872),
        ("16:9", 16 / 9, 2048, 1152),
        ("9:16", 9 / 16, 1152, 2048),
        ("21:9", 21 / 9, 2352, 1008),
        ("9:21", 9 / 21, 1008, 2352),
        ("2:1", 2 / 1, 2176, 1088),
        ("1:2", 1 / 2, 1088, 2176),
    ),
    "2K": (
        ("1:1", 1 / 1, 2048, 2048),
        ("4:3", 4 / 3, 2368, 1760),
        ("3:4", 3 / 4, 1760, 2368),
        ("5:4", 5 / 4, 2304, 1824),
        ("4:5", 4 / 5, 1824, 2304),
        ("3:2", 3 / 2, 2496, 1664),
        ("2:3", 2 / 3, 1664, 2496),
        ("16:9", 16 / 9, 2720, 1536),
        ("9:16", 9 / 16, 1536, 2720),
        ("21:9", 21 / 9, 3136, 1344),
        ("9:21", 9 / 21, 1344, 3136),
        ("2:1", 2 / 1, 2912, 1440),
        ("1:2", 1 / 2, 1440, 2912),
    ),
}


def _tensor_frame_to_pil(frame, max_side: int) -> Image.Image:
    if hasattr(frame, "detach"):
        frame = frame.detach().float().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"图像维度应为 [H,W,C]，实际为 {frame.shape}")
    frame = np.clip(frame[..., :3], 0.0, 1.0)
    pil = Image.fromarray((frame * 255.0 + 0.5).astype(np.uint8), "RGB")
    if max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
        pil = pil.resize(size, Image.Resampling.LANCZOS)
    return pil


def _as_frames(value) -> list:
    if value is None:
        return []
    if hasattr(value, "shape") and len(value.shape) == 4:
        return [value[i] for i in range(int(value.shape[0]))]
    return [value]


class LS_ImageOffset:
    """Match the input aspect ratio to a preset and resize without distortion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (["1K", "1.5K", "2K"], {"default": "1K"}),
                "resize_method": (
                    ["lanczos", "bicubic", "bilinear", "nearest"],
                    {"default": "lanczos"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING", "INT")
    RETURN_NAMES = ("image", "width", "height", "matched_ratio", "resolution_value")
    OUTPUT_TOOLTIPS = (
        "按标准尺寸等比缩放并居中裁切后的图像",
        "输出宽度",
        "输出高度",
        "匹配到的标准宽高比",
        "1K输出1024，1.5K输出1536，2K输出2048",
    )
    FUNCTION = "standardize"
    CATEGORY = "LS_Nodes/图像处理"
    DESCRIPTION = "根据输入图像的宽高比自动匹配标准1K/1.5K/2K尺寸，等比缩放后居中裁切，不拉伸图像。"

    def standardize(self, image, resolution, resize_method):
        frames = _as_frames(image)
        if not frames:
            raise ValueError("请连接需要标准化的图像。")

        first = frames[0]
        shape = tuple(first.shape)
        if len(shape) != 3:
            raise ValueError(f"图像维度应为 [H,W,C]，实际为 {shape}")
        source_height, source_width = int(shape[0]), int(shape[1])
        source_ratio = source_width / max(source_height, 1)
        ratio_name, _, target_width, target_height = min(
            STANDARD_IMAGE_SIZES[resolution],
            key=lambda item: abs(np.log(source_ratio / item[1])),
        )

        resampling = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "bilinear": Image.Resampling.BILINEAR,
            "nearest": Image.Resampling.NEAREST,
        }[resize_method]
        converted = []
        for frame in frames:
            pil = _tensor_frame_to_pil(frame, max(source_width, source_height))
            fitted = ImageOps.fit(
                pil,
                (target_width, target_height),
                method=resampling,
                centering=(0.5, 0.5),
            )
            converted.append(np.asarray(fitted, dtype=np.float32) / 255.0)

        output = image.new_tensor(np.stack(converted, axis=0))
        resolution_value = {"1K": 1024, "1.5K": 1536, "2K": 2048}[resolution]
        return (output, target_width, target_height, ratio_name, resolution_value)


NODE_CLASS_MAPPINGS = {
    "LS_ImageOffset": LS_ImageOffset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LS_ImageOffset": "LS_像素标准化",
}
