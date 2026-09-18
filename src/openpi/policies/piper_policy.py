import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

from openpi.models.stl_gate import detector

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class StuImagePreprocess(transforms.DataTransformFn):
    """Apply Piper-specific image processing before resize_with_pad."""

    stl_camera: str = "right_wrist_0_rgb"

    def __call__(self, data: dict) -> dict:
        for key in ("base_0_rgb", "right_wrist_0_rgb"):
            image = np.asarray(data["image"][key])

            # 此时 image 的格式是 HWC、RGB，通常为 uint8，
            # 并且仍然保持数据集中的原始分辨率。

            # 只处理腕部相机
            if key == self.stl_camera:
                # 门控机制 DBnet
                detections = detector.get_detector().detect(image)
                iskeyimage = detections.has_text

                # 通过门控机制，进行locate anything处理
                if iskeyimage:
                    # image_locate, bbox = locator(prompt, image)
                    pass

            image = self._process_image(image)

            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"Processed image {key} must have shape (H, W, 3), "
                    f"got {image.shape}."
                )

            # ResizeImages 使用 PIL，建议继续保持 uint8。
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

            data["image"][key] = np.ascontiguousarray(image)

        return data

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        # 示例：
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # ...
        # image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        # 或者执行裁剪、遮罩、去畸变、背景处理、颜色校正等。
        return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Convert Piper observations into the image/state layout expected by pi0 models."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # The external Piper RTC client names the head camera observation/image.
        if "observation/top_image" in data:
            base_image = data["observation/top_image"]
        elif "observation/image" in data:
            base_image = data["observation/image"]
        else:
            raise KeyError("Missing Piper head camera: expected 'observation/top_image' or 'observation/image'.")
        base_image = _parse_image(base_image)
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Return the seven continuous Piper action dimensions."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}
