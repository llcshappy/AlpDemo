# AlpDemo · EVS 鱼缸行为分析

使用 Streamlit、NumPy 和 Plotly 构建的事件视觉传感器（EVS）水族监控演示。

所有数据均由程序模拟生成，无需连接摄像头或其他硬件；页面中的评分和异常提示用于演示，不代表真实鱼类健康评估。

## 本地运行

推荐使用 Python 3.11 或更新版本。在仓库根目录执行：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows PowerShell 用户可使用 `python -m venv .venv` 创建虚拟环境，并将激活命令替换为：

```powershell
.venv\Scripts\Activate.ps1
```

启动后打开终端显示的本地访问地址。按 `Ctrl+C` 停止服务。

## 功能

- **总览仪表盘**：查看今日模拟事件总量、活跃度评分、24 小时事件趋势和异常时段。
- **月度报告**：选择月份，查看每日评分日历、评分趋势和每日明细。

模拟数据根据日期生成，同一日期的数据可以重复查看。

## 文件说明

- `app.py`：模拟数据、图表与 Streamlit 页面。
- `requirements.txt`：应用运行依赖。
- `.gitignore`：排除虚拟环境、Python 缓存和本地环境配置等文件。
