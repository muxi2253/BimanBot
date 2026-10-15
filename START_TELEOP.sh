#!/bin/bash
# 演示收集工具启动脚本

clear
echo "======================================"
echo "  双臂机器人演示收集工具"
echo "======================================"
echo ""
echo "请选择收集方式:"
echo ""
echo "1. 🖱️  鼠标拖拽教学 (最推荐！)"
echo "   - 在GUI中拖拽mocap点"
echo "   - 直观，适合复杂轨迹"
echo "   - 需要: mjpython"
echo ""
echo "2. 📊 ASCII可视化遥控"
echo "   - 终端显示俯视图"
echo "   - 键盘控制 + 位置显示"
echo "   - 适合无GUI环境"
echo ""
echo "3. ⌨️  简单键盘控制"
echo "   - 纯文本输出"
echo "   - 只有坐标数值"
echo "   - 最基础版本"
echo ""
echo "======================================"
echo ""

read -p "输入选项 (1/2/3): " choice

echo ""

case $choice in
    1)
        echo "🖱️  启动鼠标拖拽教学..."
        echo ""
        echo "使用说明:"
        echo "1. 寻找粉色/青色小球（mocap点）"
        echo "2. 鼠标左键拖拽小球"
        echo "3. 按 R 开始/停止录制"
        echo "4. 按 SPACE 重置场景"
        echo "5. 按 ESC 退出保存"
        echo ""
        echo "如果报错，请用: mjpython tools/teleop_kinesthetic.py"
        echo ""
        sleep 2
        python tools/teleop_kinesthetic.py
        ;;
    2)
        echo "📊 启动ASCII可视化遥控..."
        echo ""
        python tools/teleop_visual.py
        ;;
    3)
        echo "⌨️  启动简单键盘控制..."
        echo ""
        python tools/teleop_simple.py
        ;;
    *)
        echo "无效选项，退出"
        exit 1
        ;;
esac

echo ""
echo "收集完成！演示保存在 demonstrations/ 目录下"
