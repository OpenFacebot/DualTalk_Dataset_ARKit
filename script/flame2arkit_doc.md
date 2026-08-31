================================================================================
FLAME -> ARKit 转换原理与公式 (flame2arkit)
================================================================================

一、目标
--------------------------------------------------------------------------------
将 DualTalk 数据集提供的 FLAME2020 参数序列 (每帧 50 维表情 + 6 维姿态) 转换为
61 维的 ARKit Blendshape 运动序列，供下游说话人头像/数字人训练使用。

代码分工:
  - script/flame2arkit.py        纯数学转换 (Flame2ARKit_Linear 类，无任何文件 IO)
  - script/convert_flame2arkit.py 数据加载/保存、维度对齐、元数据路径回写
  - script/convert_flame2arkit.sh 运行入口，超参只有 metadata 目录与 output 目录

本实现以 convert_dualtalk_flame_to_arkit_bvls_csv61_1.py (基础版) 为蓝本重写，
数学上与之完全等价；增强版 (_无_1 后缀) 的头部标定扩展暂不实现，原理见第六节。


二、输入数据格式
--------------------------------------------------------------------------------
DualTalk 每个样本是一个 .npz，关键数组:
  exp  : [T, 50]  FLAME 表情系数 (仅观测到 100 维表情空间的前 50 维)
  pose : [T, 6]   pose[:, 0:3] = 颈部旋转向量 (rotvec, 弧度)
                  pose[:, 3:6] = 下颌旋转向量 (rotvec, 弧度)

内部统一拼成 FLAME106 布局 (convert_flame2arkit.py 负责对齐):
  flame106[t,   0: 50] = exp[t, 0:50]      观测的表情 50 维
  flame106[t,  50:100] = 0                 未观测的表情维度补零 (求解时忽略)
  flame106[t, 100:103] = pose[t, 0:3]      颈部 rotvec
  flame106[t, 103:106] = pose[t, 3:6]      下颌 rotvec


三、前向模型与矩阵
--------------------------------------------------------------------------------
存在一个线性前向矩阵 A (mat_final.npy, 形状 [51, 103])，把 51 个 ARKit
Blendshape 权重映射到 FLAME 的 103 维参数空间 (100 表情 + 3 下颌):

    d = a^T A,   a ∈ R^51 (ARKit 权重),  d ∈ R^103 (FLAME 增量)

矩阵列的含义:
    A[:,   0: 50]  对应表情基前 50 维 (DualTalk 观测)
    A[:,  50:100]  对应表情基后 50 维 (DualTalk 未观测, 不参与拟合)
    A[:, 100:103]  对应下颌 3 维 (由 pose[:, 3:6] 观测)
行序为 MATRIX_ARKIT51_ORDER (51 个标准 ARKit 通道; 其中 8 个 EyeLook* 通道
在 DualTalk 中无对应观测)。


四、逆向求解 (基础版核心公式)
--------------------------------------------------------------------------------
问题: 已知观测的表情 50 维与下颌 3 维，求 ARKit 权重 a。
方程数 (53) < 未知数 (43 个有效通道) 且欠定/病态，故采用带约束的正则化最小二乘。

1) 有效通道
   EyeLook* 8 个通道无观测来源，直接剔除并固定为 0;
   其余 43 个通道记为 a_act，参与求解。

2) 每帧求解一个有界线性最小二乘 (BVLS):

   min  || O a_act - y ||^2
   s.t. 0 <= a_act <= 1

   其中设计矩阵 O 与目标向量 y 由三部分堆叠而成:

   (a) 观测项 (53 行):
         O[0:50]   = A[act, 0:50]^T            y[0:50]   = exp50 (平滑后)
         O[50:53]  = A[jaw, 100:103]^T * w_jaw y[50:53]  = jaw3 * w_jaw
       w_jaw = JAW_WEIGHT = 50.0, 用于放大数值量级很小的下颌列。

   (b) 左右对称正则 (7 行, 权重 w_sym = SYMMETRY_WEIGHT = 30):
         对 7 组左右对称通道对 (如 MouthSmileLeft/Right):
         sqrt(w_sym) * (a_L - a_R) -> 0

   (c) L2 收缩正则 (43 行, 权重 w_l2 = L2_WEIGHT = 3):
         sqrt(w_l2) * a_act -> 0

   合并:  O = [ 观测 ;  sqrt(w_sym)*对称 ;  sqrt(w_l2)*I ]   (103 行 x 43 列)
          y = [ exp50_smooth ;  jaw3*w_jaw ;  0...0 ]
   用 scipy.optimize.lsq_linear(method="bvls", bounds=[0,1], tol=1e-7,
   max_iter=200) 逐帧求解, 解再裁剪到 [0,1]。

3) 时间平滑 (序列级处理, 不在数学库内)
   职责划分: flame2arkit.py 只做逐帧数学转换, 不接触任何时序信息;
   平滑由 convert_flame2arkit.py 负责, 通过超参控制开关:
       --smooth      开启 (默认, 与基础版行为一致)
       --no-smooth   关闭
   开启时对表情 50 维与颈部 3 维序列逐通道做 Savitzky-Golay 滤波
   (窗口 7, 多项式阶 3, mode="interp"), 下颌不平滑; 平滑在维度对齐之前完成。
   是否平滑会记录进 metadata 的 speaker*_smoothed 字段。

4) 关键处理: 未观测的 exp[50:100] 不当作"观测到的 0"去拟合 (否则会把解拉偏),
   仅通过 L2 正则约束整体幅度。


五、61 维运动向量的组装
--------------------------------------------------------------------------------
输出每帧 61 维 (MOTION61_ORDER):

  [ 0:51]  51 个 ARKit 权重, 按 REFERENCE_CSV_ARKIT51_ORDER 重排
  [51]     TongueOut = 0                 (FLAME 无舌头信息)
  [52:55]  HeadYaw, HeadPitch, HeadRoll  (头部旋转, 弧度)
  [55:61]  左右眼球旋转 = 0              (DualTalk 无眼球姿态)

头部旋转转换公式 (基础版):
    R   = exp(neck_rotvec^)                        (rotvec -> 旋转矩阵)
    ypr = R 的内旋 "YXZ" 欧拉角 (弧度)             (scipy Rotation.as_euler("YXZ"))
    head = clip(ypr, -1, 1)                        (逐轴裁剪到 ±1 弧度)

最终运动向量数值范围: 权重列在 [0,1], 头部三轴在 [-1,1], 其余固定为 0。
帧率 25 fps (DualTalk)。


六、头部旋转(转头)控制 (已实现, 对应原增强版能力)
--------------------------------------------------------------------------------
表情/下颌的逆向求解不受头部控制影响; 头部标定链只作用于
HeadYaw/Pitch/Roll (motion61[52:55])。标定公式:

    head_out = clip( (ypr - neutral) * signs * gains + offsets, -limits, +limits )

  - neutral: 中性(静息)姿态校正。模式 --head-center:
      none   = 不减 (默认, 基础版行为)
      first  = 减首帧姿态
      median = 减前 N 帧 (--head-calibration-frames, 默认 25) 的中位数,
               消除"头不是正着开始"的偏置。
  - signs:   --head-signs YAW PITCH ROLL, 每轴方向 ±1,
             修正坐标系手性差异 (例如 Yaw 左右相反)。
  - gains:   --head-gains, 每轴非负增益, 缩放头部动作幅度。
  - offsets: --head-offsets, 每轴常量偏移 (弧度)。
  - limits:  --head-limits, 每轴绝对值上限 (弧度), 默认 1.0 与基础版裁剪等价。

职责划分 (与 smooth 相同的原则):
  - flame2arkit.py 只做逐帧数学: convert_106_headpose 实现
        head = clip(ypr * signs * gains + offsets, ±limits)
  - convert_flame2arkit.py 负责序列级的中性姿态: compute_head_neutral 先在
    欧拉角空间算出 neutral, 再利用恒等式把它折叠进逐帧偏移:
        (ypr - neutral)*signs*gains + offsets
          == ypr*signs*gains + (offsets - neutral*signs*gains)
    即 offsets_eff = offsets - neutral*signs*gains, 数学上与增强版完全等价。
  - 参数合法性 (signs∈{±1}, gains≥0, limits>0) 由
    flame2arkit.validate_head_calibration 在任何输出写入之前校验。
  - 标定参数与实际使用的 neutral 会记录进 metadata 的
    speaker*_head_correction 字段, 便于复现。

全部使用默认值 (--head-center none, signs/gains=1 1 1, offsets=0 0 0,
limits=1 1 1) 时, 输出与基础版 _1 逐位一致 (已验证)。
与原增强版脚本的对比验证: 非默认参数 (含符号翻转/增益/偏移/限值/median 校正)
下, 头部三轴与原增强版 correct_neck_rotation 的差异 < 1e-16。


七、质量评估指标 (与基础版一致, 写入日志/元数据)
--------------------------------------------------------------------------------
  - expression50_rmse : 重建表情与观测表情的 RMSE
        rec = a^T A,  rmse = sqrt(mean( (rec[:, 0:50] - exp50)^2 ))
  - jaw3_rmse : 仅用下颌相关通道重建下颌 3 维的 RMSE
        jaw_rec = a_jaw^T A[jaw, 100:103]
  - jawopen_source_jawx_correlation : JawOpen 权重与源下颌开合的相关系数
  - upper_saturation_fraction : 权重饱和 (=1) 的比例
  - mean_left_right_difference : 7 组对称通道左右平均绝对差


八、控制开关一览 (smooth 与转头)
--------------------------------------------------------------------------------
两类控制都由 convert_flame2arkit.py 的命令行超参提供, 默认值完全复现基础版:

  平滑控制 (序列级, 表情 50 维 + 颈部 3 维, 下颌不平滑):
      --smooth       开启 Savitzky-Golay 平滑 (默认)
      --no-smooth    关闭
      记录: metadata speaker*_smoothed

  转头控制 (HeadYaw/Pitch/Roll):
      --head-center {none,first,median}     中性姿态校正模式 (默认 none)
      --head-calibration-frames N           median 模式取前 N 帧 (默认 25)
      --head-signs YAW PITCH ROLL           每轴方向 ±1 (默认 1 1 1)
      --head-gains YAW PITCH ROLL           每轴非负增益 (默认 1 1 1)
      --head-offsets YAW PITCH ROLL         每轴偏移弧度 (默认 0 0 0)
      --head-limits YAW PITCH ROLL          每轴裁剪上限弧度 (默认 1 1 1)
      记录: metadata speaker*_head_correction


九、运行方式 (含通过 convert_flame2arkit.sh)
--------------------------------------------------------------------------------
1) 直接调用 Python (单个 split):
  python3 script/convert_flame2arkit.py \
      --metadata metadata/ood.jsonl \
      --output-dir /path/to/ARKit_npy \
      [--matrix script/mat_final.npy] [--smooth | --no-smooth] \
      [--head-center median --head-gains 1.2 1.0 1.0] \
      [--overwrite] [--limit N]

2) 通过 convert_flame2arkit.sh (循环处理 train/test/ood):
   sh 脚本的超参为 METADATA_DIR 与 OUTPUT_DIR (环境变量),
   其余参数会原样透传给 convert_flame2arkit.py (脚本末尾的 "$@")。

  # 默认: 平滑开 + 头部基础行为, 输出到 DualTalk_Dataset/ARKit_npy
  bash script/convert_flame2arkit.sh

  # 指定 metadata 目录与输出目录:
  METADATA_DIR=/path/to/metadata OUTPUT_DIR=/path/to/ARKit_npy \
      bash script/convert_flame2arkit.sh

  # 关闭平滑:
  bash script/convert_flame2arkit.sh --no-smooth

  # 头部控制示例: 前 25 帧中位数校正中性姿态, Yaw 反向并放大 1.2 倍,
  # 三轴裁剪到 ±0.5 弧度:
  bash script/convert_flame2arkit.sh \
      --head-center median --head-calibration-frames 25 \
      --head-signs -1 1 1 --head-gains 1.2 1 1 \
      --head-limits 0.5 0.5 0.5

  # 只跑部分数据调试:
  bash script/convert_flame2arkit.sh --limit 2

输出: 每个样本保存 <output>/<相对路径>/<stem>.npy, 内容 [T, 61] float32;
同时把输出相对路径、源/输出文件 SHA256、帧数、smooth 开关与头部标定参数回写
到 metadata jsonl (speaker*_arkit / speaker*_smoothed / speaker*_head_correction)。
矩阵以 SHA256 校验
(f055de09c64182696499a26c2d6109349c627195bcd40c6adc3dd27f3922b34b)。
================================================================================
