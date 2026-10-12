import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import BatchEncoding


WORKSPACE = Path(__file__).resolve().parents[1]
management = types.ModuleType("comfy.model_management")
management.soft_empty_cache = Mock()
management.unload_all_models = Mock()
management.get_torch_device = Mock(return_value=torch.device("cpu"))
management.should_use_bf16 = Mock(return_value=False)
management.throw_exception_if_processing_interrupted = Mock()
comfy = types.ModuleType("comfy")
comfy.model_management = management
folder_paths = types.ModuleType("folder_paths")
spec = importlib.util.spec_from_file_location("ls_multi_test", WORKSPACE / "LS_LLM_MultiImageEdit.py")
node_module = importlib.util.module_from_spec(spec)
stubs = {"comfy": comfy, "comfy.model_management": management, "folder_paths": folder_paths}
original_modules = {name: sys.modules.get(name) for name in stubs}
sys.modules.update(stubs)
try:
    spec.loader.exec_module(node_module)
finally:
    for name, original in original_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class MultiImageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folder_paths.models_dir = self.directory.name
        model_dir = Path(self.directory.name) / "prompt_generator" / node_module.MODEL_NAME
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        for name in ("soft_empty_cache", "unload_all_models", "get_torch_device", "should_use_bf16", "throw_exception_if_processing_interrupted"):
            getattr(management, name).reset_mock()
        management.throw_exception_if_processing_interrupted.side_effect = None
        self.node = node_module.LS_LLM_MultiImageEdit()
        self.params = {
            "role_positioning": "视觉分析师",
            "user_instruction": "用图2的背景替换图1的背景。",
            "model": node_module.MODEL_NAME,
            "output_language": "中文",
            "keep_model_loaded": False,
            "release_comfy_vram": True,
            "temperature": 0.7,
            "top_p": 0.85,
            "max_new_tokens": 128,
            "min_pixels": 4096,
            "max_pixels": 16384,
            "video_sample_frames": 3,
            "seed": 8888,
            "attention": "eager",
        }
        self.processor = Mock()
        self.processor.apply_chat_template.return_value = "formatted prompt"
        self.processor.side_effect = lambda **kwargs: BatchEncoding({"input_ids": torch.tensor([[10, 20, 30]])})
        self.processor.batch_decode.return_value = ["最终编辑提示词"]
        self.model = Mock()
        self.model.generate.return_value = torch.tensor([[10, 20, 30, 40, 50]])
        self.processor_loader = self.enterContext(patch.object(node_module.AutoProcessor, "from_pretrained", return_value=self.processor))
        self.model_loader = self.enterContext(patch.object(node_module.Qwen3VLForConditionalGeneration, "from_pretrained", return_value=self.model))

    def test_ten_ports_batch_and_video_keep_order(self):
        ports = self.node.INPUT_TYPES()["optional"]
        self.assertEqual(list(ports), [f"image_{i}" for i in range(1, 11)] + ["video"])
        media = {f"image_{i}": torch.full((1, 4, 4, 3), i / 10) for i in range(1, 11)}
        media["image_3"] = torch.cat([media["image_3"], torch.full((1, 4, 4, 3), 0.35)])
        media["video"] = torch.stack([torch.full((4, 4, 3), i / 4) for i in range(5)])
        messages, images = node_module._build_messages("role", "instruction", "中文", 3, media)
        labels = [part["text"] for part in messages[1]["content"] if part["type"] == "text"]
        self.assertEqual(len(images), 14)
        self.assertIn("参考图3（image_3），批次第2/2张", labels)
        self.assertEqual(labels[-4:-1], [
            "视频顺序帧1/3（原始第1帧）", "视频顺序帧2/3（原始第3帧）", "视频顺序帧3/3（原始第5帧）",
        ])
        self.assertEqual([im.getpixel((0, 0))[0] for im in images[-3:]], [0, 128, 255])
        self.assertEqual(labels[-1], "instruction")

    def test_text_only_decode_and_release(self):
        result = self.node.generate(**self.params)
        self.assertEqual(result["result"], ("最终编辑提示词",))
        self.assertIsNone(self.processor.call_args.kwargs["images"])
        decoded_ids = self.processor.batch_decode.call_args.args[0]
        self.assertEqual(decoded_ids.tolist(), [[40, 50]])
        self.assertIsNone(self.node.model)
        self.assertIsNone(self.node.processor)
        management.unload_all_models.assert_called_once()
        self.assertTrue(self.model_loader.call_args.kwargs["local_files_only"])
        self.assertNotIn("quantization_config", self.model_loader.call_args.kwargs)

    def test_kept_model_uses_new_pixel_limits_and_reloads_attention(self):
        self.params["keep_model_loaded"] = True
        self.params["image_1"] = torch.zeros((1, 4, 4, 3))
        self.node.generate(**self.params)
        self.params["min_pixels"] = 16384
        self.params["max_pixels"] = 65536
        self.node.generate(**self.params)
        self.assertEqual(self.model_loader.call_count, 1)
        self.assertEqual(self.processor.call_args.kwargs["min_pixels"], 16384)
        self.assertEqual(self.processor.call_args.kwargs["max_pixels"], 65536)
        self.params["attention"] = "sdpa"
        self.node.generate(**self.params)
        self.assertEqual(self.model_loader.call_count, 2)
        self.assertEqual(self.model_loader.call_args.kwargs["attn_implementation"], "sdpa")

    def test_zero_temperature_and_sampling_controls(self):
        self.params["temperature"] = 0
        self.node.generate(**self.params)
        settings = self.model.generate.call_args.kwargs
        self.assertFalse(settings["do_sample"])
        self.assertNotIn("temperature", settings)
        self.assertNotIn("top_p", settings)
        self.params["temperature"] = 0.4
        self.node.generate(**self.params)
        settings = self.model.generate.call_args.kwargs
        self.assertTrue(settings["do_sample"])
        self.assertEqual(settings["temperature"], 0.4)
        self.assertEqual(settings["top_p"], 0.85)

    def test_inference_failure_releases_kept_model(self):
        self.params["keep_model_loaded"] = True
        self.model.generate.side_effect = RuntimeError("simulated inference failure")
        with self.assertRaisesRegex(RuntimeError, "simulated inference failure"):
            self.node.generate(**self.params)
        self.assertIsNone(self.node.model)
        self.assertIsNone(self.node.processor)
        self.assertIsNone(self.node.loaded_settings)

    def test_partial_load_failure_releases_processor(self):
        self.params["keep_model_loaded"] = True
        self.model_loader.side_effect = RuntimeError("simulated load failure")
        with self.assertRaisesRegex(RuntimeError, "simulated load failure"):
            self.node.generate(**self.params)
        self.assertIsNone(self.node.processor)
        self.assertIsNone(self.node.loaded_settings)

    def test_cancel_during_generation_releases_model(self):
        self.params["keep_model_loaded"] = True
        def interrupting_generate(**kwargs):
            management.throw_exception_if_processing_interrupted.side_effect = RuntimeError("cancelled")
            kwargs["stopping_criteria"][0](kwargs["input_ids"], None)
        self.model.generate.side_effect = interrupting_generate
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.node.generate(**self.params)
        self.assertIsNone(self.node.model)

    def test_model_combo_cannot_be_used_as_a_path(self):
        self.params["model"] = "../../another_model"
        with self.assertRaises(ValueError):
            self.node.generate(**self.params)
        self.model_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
