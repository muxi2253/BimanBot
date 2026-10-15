#!/bin/bash
################################################################################
# 一键启动快速测试
# 包含所有优化和检查
################################################################################

echo "🚀 快速实验启动程序"
echo "===================="
echo ""

# 检查电量
battery=$(pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%')
echo "📱 当前电量: ${battery}%"

if [ "$battery" -lt 40 ]; then
    echo "⚠️  警告: 电量较低 (${battery}%)"
    echo "   预计需要2-3小时，可能电量不足"
    read -p "   确定继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "📋 运行前提醒:"
echo "   1. 降低屏幕亮度"
echo "   2. 关闭浏览器和聊天软件"
echo "   3. 开启低电量模式"
echo ""
read -p "准备好了吗? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

echo ""
echo "✅ 开始实验..."
echo "   预计时间: 2-3小时"
echo "   预计完成: $(date -v+3H '+%H:%M')"
echo ""

# 记录开始时间
start_time=$(date +%s)
start_battery=$battery
echo "开始时间: $(date)" > experiment_status.txt
echo "开始电量: ${battery}%" >> experiment_status.txt

# 运行实验（使用caffeinate防止休眠）
caffeinate -i ./experiments/run_quick_test.sh

# 记录结束时间
end_time=$(date +%s)
end_battery=$(pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%')
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))

echo "" | tee -a experiment_status.txt
echo "结束时间: $(date)" | tee -a experiment_status.txt
echo "结束电量: ${end_battery}%" | tee -a experiment_status.txt
echo "总耗时: ${hours}h ${minutes}m" | tee -a experiment_status.txt
echo "电量消耗: $((start_battery - end_battery))%" | tee -a experiment_status.txt
echo ""
echo "✅ 实验完成！"
echo ""
echo "下一步:"
echo "  1. 分析结果: python experiments/analyze_results.py"
echo "  2. 查看图表: open experiments/results/learning_curves.png"
echo ""
