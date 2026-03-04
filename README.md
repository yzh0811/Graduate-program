# 基于 AI Agent 的股票组合投资推荐系统（原型）

本仓库是毕业设计原型工程骨架：**多智能体（LangChain + LangGraph）**驱动的 A 股组合推荐，支持**月度回测**与**Web 可视化**。

## 目录结构

- `backend/`: FastAPI 后端（同时提供网页 UI）
  - `app/agents/`: 多智能体图（LangGraph）与提示词
  - `app/data/`: 数据提供方（AkShare/Tushare）与缓存
  - `app/backtest/`: 回测引擎与绩效指标
  - `app/templates/` + `app/static/`: 前端页面（无需 Node）

## 快速开始（Windows / PowerShell）

进入后端目录并创建虚拟环境：

```powershell
cd d:\毕设\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy env.example .env
```

启动服务：

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

打开浏览器访问：

- `http://127.0.0.1:8000/`

## 下一步你需要补的内容

- 在 `.env` 里填写你要使用的大模型配置（OpenAI 兼容 / Ollama 等）
- 根据你的偏好选择数据源（Tushare/AkShare），并完善 `app/data/providers.py`
- 把回测区间、基金对比、双模型横向对比做成可配置的实验脚本/页面

