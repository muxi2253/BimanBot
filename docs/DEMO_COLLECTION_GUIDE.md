# 演示收集方案对比

## 三种收集方式

### 方案1: 🖱️ 鼠标拖拽运动学教学 ⭐⭐⭐⭐⭐

**最推荐！最直观！**

```bash
./START_TELEOP.sh
# 选择 1
```

**优点：**
- ✅ 完全可视化GUI
- ✅ 鼠标直接拖拽末端执行器
- ✅ 实时看到机器人响应
- ✅ 适合复杂轨迹

**操作：**
1. GUI自动打开
2. 找到**粉色/青色小球**（mocap点）
3. **鼠标左键拖拽**小球移动
4. 机器人末端跟随
5. 按 `R` 开始/停止录制
6. 拖拽完整轨迹后再按 `R` 停止

**问题排查：**
如果报错 "requires mjpython"，使用：
```bash
mjpython tools/teleop_kinesthetic.py
```

**数据保存：**
```
demonstrations/dual_arm_kinesthetic/kinesthetic_demos_*.pkl
```

---

### 方案2: 📊 ASCII可视化遥控 ⭐⭐⭐⭐

**无GUI环境最佳选择**

```bash
./START_TELEOP.sh
# 选择 2
```

**优点：**
- ✅ 终端显示俯视图（ASCII图形）
- ✅ 实时位置、距离、高度信息
- ✅ 清楚知道何时可以抓取
- ✅ 不需要GUI

**界面示例：**
```
====================================================
   俯视图 (X-Y平面)  近 ↑ 远
====================================================
│                                                  │
│+                                               + │  ← 机器人TCP
│                                                  │
│  ●                                          ◆    │  ← 物体
│                                                  │
│       ░░░░░                                      │  ← 目标箱
│       ░░░░░                                      │
│       ░░░░░                                      │
│                                                  │
====================================================

高度(Z轴):
  左臂: 1.80m  左物体: 0.15m
  右臂: 1.80m  右物体: 0.15m

距离:
  左臂→左物体: 1.663m 
  右臂→右物体: 1.657m 
```

**图例：**
- `+` 左臂TCP（开）
- `✖` 左臂TCP（闭）
- `●` 左物体（红色）
- `◆` 右物体（绿色）
- `░` 目标箱

**操作：**
1. 输入控制命令（可组合）
2. 观察俯视图变化
3. 当距离 < 0.08m 时会提示 "✓ 可抓取"
4. 按 `r` 开始/停止录制

**控制示例：**
```bash
命令 > w         # 左臂前进
命令 > qq        # 左臂下降（连续两次）
命令 > x         # 关闭左夹爪（抓取）
命令 > eee       # 左臂上升
命令 > sss       # 后退到目标
命令 > qq        # 下降
命令 > z         # 打开夹爪（放置）
```

**数据保存：**
```
demonstrations/dual_arm_teleop/teleop_demos_*.pkl
```

---

### 方案3: ⌨️ 简单键盘控制 ⭐⭐

**最基础版本**

```bash
./START_TELEOP.sh
# 选择 3
```

**优点：**
- ✅ 最简单
- ✅ 纯文本输出
- ✅ 低资源消耗

**缺点：**
- ❌ 只有坐标数值
- ❌ 难以判断位置关系
- ❌ 需要记忆坐标

**适用场景：**
- 已经熟悉坐标系
- 调试/快速测试
- SSH远程操作

**数据保存：**
```
demonstrations/dual_arm_teleop/teleop_demos_*.pkl
```

---

## 推荐使用流程

### 新手：方案2 (ASCII可视化)

1. **启动工具**
   ```bash
   ./START_TELEOP.sh
   # 选择 2
   ```

2. **先熟悉控制**（不录制）
   ```bash
   命令 > w          # 尝试前进
   命令 > q          # 尝试下降
   ```
   观察俯视图的 `+` 符号移动

3. **简化任务：只控制左臂**
   ```bash
   命令 > r          # 开始录制 🔴
   
   # 移动到左物体 ●
   命令 > www        # 前进
   命令 > qq         # 下降
   
   # 当距离显示 "✓ 可抓取" 时
   命令 > x          # 关闭夹爪
   
   # 提升
   命令 > eee
   
   # 移动到目标 ░
   命令 > sss
   命令 > qq
   
   # 放置
   命令 > z          # 打开夹爪
   
   命令 > r          # 停止录制 ⚪
   ```

4. **重复10-20次**
   ```bash
   命令 > reset      # 每次完成后重置
   命令 > r          # 开始新一轮录制
   ...
   ```

5. **退出保存**
   ```bash
   命令 > quit
   ```

### 进阶：方案1 (鼠标拖拽)

如果您有GUI访问权限：

1. 启动工具
2. 在3D窗口中找到**小球**（mocap点）
3. **鼠标拖拽**小球到物体上方
4. 按 `R` 开始录制
5. 继续拖拽完成完整轨迹
6. 按 `R` 停止录制
7. 重复多次

---

## 键位总结

### 左臂控制
- **W/S** - X轴 (前/后)
- **A/D** - Y轴 (左/右)  
- **Q/E** - Z轴 (下/上)
- **Z/X** - 夹爪 (开/关)

### 右臂控制
- **I/K** - X轴 (前/后)
- **J/L** - Y轴 (左/右)
- **U/O** - Z轴 (下/上)
- **N/M** - 夹爪 (开/关)

### 录制控制
- **R** - 开始/停止录制
- **reset** - 重置环境
- **quit** - 退出保存

---

## 数据检查

收集完成后，检查数据：

```bash
python tools/collect_demonstrations.py --inspect demonstrations/dual_arm_teleop/teleop_demos_*.pkl
```

或：

```bash
python tools/collect_demonstrations.py --inspect demonstrations/dual_arm_kinesthetic/kinesthetic_demos_*.pkl
```

---

## 常见问题

### Q: 方案1报错 "requires mjpython"？

macOS需要用特殊的Python解释器：
```bash
mjpython tools/teleop_kinesthetic.py
```

### Q: 看不到mocap小球？

在MuJoCo viewer中：
1. 右键打开选项菜单
2. 勾选 "Visualization" → "mocap"
3. 小球会显示为半透明的粉色/青色球体

### Q: 方案2的俯视图不准确？

俯视图是简化的2D投影，主要用于：
- 判断左右位置
- 判断前后距离
- 结合下方的"高度(Z轴)"信息判断上下

### Q: 要录多少条演示？

建议：
- **最少10条** - 快速测试BC是否能学习
- **推荐20-50条** - 稳定训练
- **理想100+条** - 高性能策略

即使部分演示失败也保存，数据多样性很重要！

### Q: 单臂还是双臂？

**新手先单臂**：
- 只控制左臂完成左物体的抓取-放置
- 完全忽略右臂
- 更容易成功，快速获得有效数据

**熟练后再双臂**：
- 同时控制两臂
- 需要协调配合
- 但效率更高

---

## 下一步

收集到演示后：

```bash
# 1. 检查数据质量
python -c "
import pickle
with open('demonstrations/dual_arm_teleop/teleop_demos_*.pkl', 'rb') as f:
    demos = pickle.load(f)
    print(f'轨迹数: {len(demos)}')
    print(f'平均长度: {sum(d[\"episode_length\"] for d in demos) / len(demos):.1f}')
"

# 2. 训练BC策略（待实现）
python train_bc_pickplace.py

# 3. BC+RL结合训练（待实现）
python train_bc_rl.py --bc-init
```
