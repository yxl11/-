# 巡查任务比较的独立复现包

本包对应本文采用的 P：`kc=0.24, kf=0.16`；点位编队 RAS04：`alpha=coupling=0.4`；核分布匹配 MMD19：`gamma=20`。三组初态为33001、33002、33003。内部历史方法名`RAS08`指Ren–Sorensen方法，不是0.8增益。

本包直接复制既有资料，不使用硬链接；原始研究工作区不作为运行时依赖。所有运行说明均从本目录执行，目录可整体复制到其他位置。Windows深层路径由入口自动加长路径前缀；建议解压到较短路径。源码、数据、协议保持原字节，清单见`manifest.json`。

## 1. 环境

已在Python 3.11.6、Windows及`requirements.txt`中的NumPy/SciPy/Matplotlib版本完成下列验收。其他系统未执行测试，不承诺跨BLAS逐位一致；浮点比较允许`1e-9`或图数组`3e-10`误差。

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -X utf8 -B verify_package.py
```

已有匹配环境可直接使用`python`。`environment.json`记录本包复算环境；本轮计时机器与计时环境另见`timing/current/environment.json`，不能将其冒充历史计时的硬件记录。

## 2. 只复算已有结果（推荐）

```powershell
python -X utf8 -B verify_package.py
python -X utf8 -B reproduce.py --out my_results
```

`my_results`必须是不存在的新目录；入口拒绝覆盖已有结果和`data/snapshots/reference/timing`。已完成的一次验收输出放在`recomputed/`，所以请勿以它作为新输出目录。`reproduce.py`不会加载闭环运行器、推进平台状态或重新调参。

输出包含：

- `aggregates.csv/results.json`：0–8 s、8–86 s、86–178 s全部三方法统计；主表、初始部署表及执行表可直接复算。
- `per_seed.csv/role_per_seed.csv`：逐初态、逐角色与任务边统计。
- `comparisons.csv`：P对两对照的绝对差、相对变化及三初态方向。
- `completion_terminal.csv`：环巡完成数和178 s终端在域数。
- `figure2_plot_data.npz`：第一初态的群心/目标航迹、8/16/24 s固定翼快照、178 s末态、目标形状与布局坐标；绘图布局范围沿用原图，其他数据从本包轨迹重建。
- `figure3_plot_data.npz`：九条原始群级误差曲线及三初态逐时中位数。该中位数与主表的“先每初态RMS、再三初态平均”不是同一运算。
- `report.json/numerical_checks.csv`：与已独立核验的原结果和两图数组逐项比对。

积分采用原冻结`frozen_analysis/metrics.py`。Ec/EB由历史位置与解析参考重建；密度、形状和区域记录来自已审计的fast/slow指标文件，并按原窗口定义积分。本入口复现原报告口径，不更换KDE估计器。群心峰值取全部初态和时刻的最大值，距离下界取全部初态和执行段的最小值；其余按每条轨迹统计后取三初态均值。

每个数据目录包括history、fast/slow指标、配置、初态、控制身份、summary、原输入清单及终端证书。为控制体积，未复制大型逐候选事件与证书ledger日志；因此本入口不声称独立逐段重审所有历史认证，既有完整评价审计记录见`reference/audit/evaluation_audit.json`。

## 3. 原完整闭环入口：默认只预检

```powershell
python -X utf8 -B run_full.py --method P --seed 33001
python -X utf8 -B run_full.py --method RAS04 --seed 33001
python -X utf8 -B run_full.py --method MMD19 --seed 33001
```

支持33001、33002、33003。默认仅校验原冻结来源、选择记录、初态和点位分配，构造控制器并计算一个初始名义指令；不创建闭环运行目录、不调用平台推进或认证搜索。

完整178 s复跑必须明确添加`--execute`，例如：

```powershell
python -X utf8 -B run_full.py --method P --seed 33001 --execute --work-dir C:\repro_runs\P_s33001
```

工作目录必须不存在。入口先把相应冻结快照复制到全新的工作目录，再调用未经修改的原运行器；运行输出只写入这份工作副本。P调用原`run_candidate.py --stage evaluation --candidate B`，对照调用原paired `worker.py`。参数、时钟、局部修正及持续后备均保持原协议。九条完整运行耗时远高于图表复算；本次交付**没有执行这些完整复跑**。

### 为什么有两个快照

- `snapshots/P`恢复P校准后评价所用主稿SHA256 `195ab437300ab44339dfda33f169d41a2984a6315550a6cc367dfa46cb8091fd`。
- `snapshots/external`恢复原对照运行记录中的主稿SHA256 `9b1aae95ce0620415e95be64f79067cc7e48a2b0a1d13f613558437fa376d105`。

恢复的是已找到的精确历史归档，不修改协议中的任何旧哈希。两份都保存原来的`output/reviews`和`output/simulations`布局，避免当前论文正文修改导致历史输入校验失败。它们保存历史实现；后续论文对接口输入域的澄清不被悄悄写回旧轨迹实现。现有数据位于独立`data/`，不能被完整复跑覆盖。

## 4. 计时记录：只核验，不重测

`timing/current/`为本轮22个固定状态、每法一次预热及三次计时的既有记录，包含CPU i7-14650HX及库版本、状态包、协议、逐次记录和检查结果。`timing/historical/`保留此前计时。执行`verify_package.py`会校验这些文件哈希，不会重新计时。

本轮三方法各有18/66次单候选认证超过20 ms；闭环轨迹没有注入CPU计算延迟。计时范围为单次command以及另行fresh candidate认证，限速位于两次计时之外，不包括完整局部搜索、保留后备执行或平台推进。目标点生成时间来自原记录；分配时间按相应计时记录身份区分。

## 5. 验收与来源

`reference/density_reference/`另行归档同人数、同带宽的独立目标采样解析参照，包括原解析脚本、`results.json`、`validation.json`、逐时与窗口CSV、NPZ数据、`source_manifest.json`和报告。这是独立解析计算，不属于九条闭环轨迹；本次仅复制既有文件并登记哈希，没有重新计算。`reproduce.py`不会自动执行该脚本；归档脚本保留原作者端来源路径，供审查计算定义。其原验算报告为PASS，复制核验见`reports/density_reference_copy_check.json`。

- `reports/delivery_validation.json`：移动副本后从其他工作目录运行，全部九组合完整入口预检、原初始指令比对、来源再校验。
- `reports/relocated_reproduction_report.json`：移动副本中351项逐初态指标与37组图数组复算比对。
- `manifest.json`：复制的每个文件、原工作区相对来源、SHA256与字节数。来源路径仅作溯源，不在运行时回读原工作区。
- `entrypoint_hashes.json`：交付入口自身哈希。
- `reports/manuscript_table_check.json`：由复算结果生成27行表格文字，逐行核对当前稿六张轨迹统计表。
- `reports/provenance_check.json`：逐项确认打包来源仍与原始文件一致。
- `reference/manuscript_before_revision.tex`：本次两轮修订前的正文，保留原主表作核对；后续文字和计时更新以正式论文为准。

`build_package.py`、`complete_dependencies.py`、`add_density_reference.py`和`finalize_delivery.py`仅记录作者端如何构建及验收本包，不是使用者运行入口。`validate_delivery.py`用于作者端复制验收，可能额外占用约240 MiB。正常使用只需`verify_package.py`、`reproduce.py`及按需使用`run_full.py`。冗余的`migration_probe/`仅为已完成的迁移验收副本；清理时自动审批拦截了删除操作，现予保留，分发本包时应排除该目录。

建议随论文提交整个目录作为补充复现材料。目前本包为本地交付，没有虚构的公开URL或DOI；可在提交补充材料后按实际发布位置更新论文入口。
