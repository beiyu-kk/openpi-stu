# LocateAnything 源码来源

本目录接入的是 `test_single_image.py` 实际执行的推理源码依赖链。
模型仓库中的实现已经成为 `openpi.models.stl.locate_anything` 的本地源码，
正常推理不需要原 LocateAnything 仓库、外部 Python 搜索路径或独立服务。

## 来源

- Worker 与原始 CLI：`/home/ubun/project/stu_vla/locate-anything/locateanything_worker.py`、`scripts/test_single_image.py`。
- 模型、预处理、解码、批处理和注意力实现：`/media/ubun/16T/checkpoints/stuvla/locateanything/`。
- 模型标识：`nvidia/LocateAnything-3B`。
- 本地快照没有可用的 Git revision；原始文件 SHA256 记录在 `upstream_manifest.json`。
- 原始版权头保留，发布包附带的许可证原文保存在 `locate_anything/LICENSE`。
  不将这些文件重新声明为 OpenPI 的许可证。

## 保留的实现

`modeling_locateanything.py`、`modeling_qwen2.py`、`modeling_vit.py`、
两份模型配置、两份预处理实现、`generate_utils.py`、SDPA/Magi mask、
`batch_utils` 与 `kernel_utils` 均在本项目内。
MTP/NTP/hybrid 解码、任务 prompt、坐标解析和绘图规则保持原行为。

没有接入独立的 LocateAnything 数据集训练工程和评测数据。
发布包的 `batch_infer.py` 是应用入口，不是单图模型依赖，已由本地 Worker/API 入口覆盖。

## 本地适配

1. 增加 `loader.py`，显式构造本地模型与 processor，不执行权重目录的 Python 文件。
   tokenizer 使用 Transformers 内置类，关闭 `trust_remote_code`。
2. 完整读取 checkpoint 的 processor 配置及聊天模板。
   该 checkpoint 的 `in_token_limit=25600`，不能误用类默认的 4096。
3. 在模型构造前分别设置文本和视觉 attention，使 OpenPI 的 Transformers 4.53.2 使用正确的 SDPA 路径。
   没有修改权重名称、模型结构或采样算法。
4. 去除 image processor 的全局 AutoImageProcessor 注册，避免影响同进程其他模型。
5. LMDB/OpenCV、Decord 改为在对应数据读取函数中导入，单图 PIL 输入无需安装这些后端。
6. Worker 统一通过本地 loader 加载，保留原有任务、视觉提示和批处理接口；新增 `SceneTextLocator` 结构化接口。
7. 批处理使用相对包导入、显式模型/设备配置；调度器读取当前设备，Worker 不再写入 `LA_FLASH_*` 环境变量。
   原运行时仍是单进程单模型配置，初始化第二个不同配置会明确报错。
8. 原单图 CLI 的调度、解析、绘图拆入 STL 内部模块；新增随机种子参数和离线参数，默认生成设置不变。

批处理与可选加速源文件保留原风格，未做整体格式化或算法重写。
`verify.py` 中的外部源码导入仅用于显式选择的 `--implementation original` 对照流程。
