# Windows → Git远程仓库 → Ubuntu 20.04 / ROS 1

迁移准备日期：2026-09-15。目标远程仓库为`https://github.com/yxl11/-`，本地已配置`origin`，原始分支为`master`。以下提交姓名/邮箱占位内容仅在本机尚未配置身份时填写。上传是否完成以`git status`、远程分支和推送结果为准。

## 迁移包含的内容

Git保留Python代码、配置、依赖版本、测试、方法说明、申请书提取文本/公式图片、正式仿真结果、88秒三机轨迹和演示视频。`.gitignore`排除Windows虚拟环境、缓存、中间调试/冒烟运行，以及将来的ROS编译产物和飞行录包；被忽略的现有文件仍留在本机，没有删除。

原始`面上项目-正文-0310.docx`在Windows桌面，位于项目目录之外，不会被本次Git上传自动包含。`analysis/`中已有提取内容；如需保留Word原件，应另行复制或传输。申请书原文与项目方法属于研究材料，建议新建私有仓库。

本次保留正式原始日志以便在Ubuntu重新绘图和审计。后续大量实验产生的新日志可使用独立数据归档，不必把每次运行全部提交。当前单个文件未达到GitHub普通Git的100 MiB阻止阈值，无需仅为首次迁移引入Git LFS。[GitHub文件大小说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

`.gitattributes`保留现有源文件与数据的原始换行字节，避免Windows的自动换行转换导致历史SHA256校验变化。新增`.sh`脚本统一使用LF。

## 1. 创建远程空仓库

在GitHub、Gitee或已有Git服务中创建一个**私有空仓库**。仓库名可选`cooperative-sling-transport`。不勾选自动生成README、.gitignore或License，因为本地已准备README和其他项目文件。

本项目已选定仓库`https://github.com/yxl11/-`，以下使用其HTTPS克隆地址`https://github.com/yxl11/-.git`。仓库名称只有一个连字符，因此Ubuntu克隆时显式指定本地目录`cooperative-sling-transport`。

## 2. Windows首次提交和上传

在PowerShell进入本地项目目录：

```powershell
Set-Location -LiteralPath 'C:\Users\31977\Documents\ChatGPT\协同吊运实验'
git config user.name "你的提交姓名"
git config user.email "你的提交邮箱"
git add .
git diff --cached --stat
git commit -m "Import cooperative sling transport simulation and replay experiment"
git remote -v
git push -u origin master
```

这里的姓名/邮箱只写入当前仓库配置，不修改全局身份。首次推送按平台的登录方式认证；私有仓库的Ubuntu拉取也需要同一账号授权或已配置的SSH密钥。认证信息不写入项目文件。

本机`origin`已配置，无需重复添加。如果在另一份没有remote的本地副本上传，可执行`git remote add origin https://github.com/yxl11/-.git`。如果远程已有README或其他提交，应先读取远程历史并处理合并，而不是直接强推。本项目当前不需要重新`git init`或修改分支名称。

当前Windows系统使用本机代理`127.0.0.1:7890`，Git直连GitHub曾超时。需要时仅对此次命令使用系统已有代理：`git -c http.proxy=http://127.0.0.1:7890 push -u origin master`。该地址只适用于这台Windows电脑，不应复制为Ubuntu的代理设置。

流程依据：[GitHub导入本地项目](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。

## 3. Ubuntu拉取代码

在Ubuntu终端执行，使用实际仓库地址：

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/yxl11/-.git cooperative-sling-transport
cd cooperative-sling-transport
git status
```

建议先放在普通项目目录，不直接嵌入现有PX4源码或catkin工作空间。本仓库目前是Python研究仿真工程，还不是ROS包。

## 4. 为Python仿真单独创建环境

本项目Windows验证环境是Python 3.12.14；Ubuntu端使用独立Python 3.12环境。不要复制Windows的`.venv`，也不要为了本项目替换`/usr/bin/python3`或修改现有ROS的Python环境。虚拟环境应在目标机器重建。[Python虚拟环境说明](https://docs.python.org/3/library/venv.html)

以下使用uv安装独立解释器，并在当前项目建立`.venv`。若已安装uv，跳过安装uv的两条命令；若已具备独立Python 3.12，也可直接用它创建虚拟环境。

```bash
sudo apt update
sudo apt install -y curl ca-certificates fonts-noto-cjk
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/sling-uv-install.sh
sh /tmp/sling-uv-install.sh
"$HOME/.local/bin/uv" python install 3.12
"$HOME/.local/bin/uv" venv --python 3.12 .venv
"$HOME/.local/bin/uv" pip install --python .venv/bin/python -r requirements-lock.txt
```

如uv使用了自定义安装位置，请用实际可执行文件路径替换。uv在用户目录安装解释器，这里没有使用`--default`替换无版本的python命令，也没有用`sudo pip`向系统环境安装依赖。仿真命令始终显式调用`.venv/bin/python`，无需在ROS终端中激活它。

依赖锁文件记录当前验证版本；Linux会安装对应平台的软件包，而非Windows二进制包。目标Ubuntu尚未实测，若某个锁定版本无法安装，应保留错误并针对该依赖处理，不要直接大范围升级ROS或系统Python。[uv安装说明](https://docs.astral.sh/uv/getting-started/installation/)、[独立Python安装说明](https://docs.astral.sh/uv/guides/install-python/)

`matplotlibrc`在项目根目录设置Linux中文字体。运行绘图命令时保持当前目录为项目根目录，已安装的`fonts-noto-cjk`用于中文图表。

## 5. 验证迁移结果

先检查Python版本、运行36项单元测试，再执行2秒短仿真。短仿真只确认依赖和入口可运行，不代表完整任务验收通过。

```bash
.venv/bin/python --version
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_simulation.py --scenario nominal --duration 2 --output results/ubuntu_smoke
```

仓库保留已有正式结果，可以直接重绘图表：

```bash
.venv/bin/python tools/render_experiment.py
```

需要重新生成动画时加`--animation`。重新运行整套实验建议输出到新目录，保留迁移前的数据：

```bash
.venv/bin/python run_simulation.py --scenario nominal --leg-duration 32 --output results/experiment_ubuntu/planning
.venv/bin/python run_replay.py --source results/experiment_ubuntu/planning --output results/experiment_ubuntu --suite --workers 3
.venv/bin/python run_replay.py --source results/experiment_ubuntu/planning --output results/experiment_ubuntu --cases configs/replay_capture_cases.json --append --workers 3
.venv/bin/python tools/render_experiment.py --root results/experiment_ubuntu --animation
.venv/bin/python -m unittest discover -s tests -v > results/experiment_ubuntu/unit_tests.txt 2>&1
.venv/bin/python tools/report_experiment.py --root results/experiment_ubuntu
```

广范围`--suite`包含预计不能满足末段1°倾角要求的工况，会保存结果并返回非零状态；这不等同于程序崩溃。应查看各工况`metrics.json`后继续候选拍摄工况。复现命令应逐条执行，不要用`&&`把此预期非零返回与后续步骤绑定。

## 6. 在Ubuntu继续开发Gazebo / ROS 1

当前已有环境是Ubuntu 20.04、ROS 1、Gazebo和PX4。保留该路线，先查清实际版本，不按此前ROS 2 / Gazebo Harmonic建议直接升级：

```bash
lsb_release -ds
rosversion -d
gazebo --version
```

进入实际PX4源码目录后，运行`git describe --tags --always`和`git status --short`记录版本与本地改动。ROS指令在原有ROS终端执行，使用其原有系统Python和环境。

后续研究仿真继续用Python 3.12；ROS 1接口应在ROS自身环境编译和运行。两边通过冻结轨迹CSV、日志或明确定义的进程间接口交换数据，不把Python 3.12虚拟环境直接当作ROS 1运行环境。

可把下面这段任务说明交给Ubuntu端的Codex：

> 请先阅读README.md、docs/MIGRATION_UBUNTU20_ROS1.md、docs/force_prediction_mapping.md和results/experiment_v1/轨迹回放可行性报告.md。这个项目已实现三机吊载物理模型、申请书PDE编队、ESO与平动Tube MPC合力预测、三绳向量力分配和编队逆映射，以及88秒冻结轨迹回放。接下来使用当前Ubuntu 20.04、ROS 1、Gazebo和PX4环境，先检查实际版本与现有改动，验证Python项目迁移，再开发物理引擎接入。不要擅自改成ROS 2或升级现有系统。研究算法保留在离线规划中，执行端首先复现三机轨迹；不要把负载/绳力真值偷送进回放控制器，也不要用直接设置模型位置替代物理运动。用户不要求算法性能对照。

桌面端对话不会随Git仓库自动搬到Ubuntu；上述文档和任务说明用于交接上下文。当前并未实现Gazebo或PX4运行接口，不能把已保存Python仿真指标当作Gazebo或实机指标。

## 7. 两台电脑后续同步

每次开始工作先在工作树干净时运行`git pull --ff-only`。完成修改后`git add`、`git commit`、`git push`。切换电脑前先推送，另一台再拉取；遇到未提交改动或分叉时先检查并处理，不用强推覆盖另一台工作。
