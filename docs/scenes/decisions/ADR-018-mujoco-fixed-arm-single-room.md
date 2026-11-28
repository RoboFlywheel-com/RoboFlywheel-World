# ADR-018：MuJoCo 固定机械臂与单房优先

## Context

2026-10-04用户明确改用MuJoCo，指出没有移动平台，先做好单房摆放/安装。

## Evidence

SPEC-033/034只有固定Piper的PyBullet局部基线，不能提供跨房导航。
本机MuJoCo 3.12.0真实导入已保存Piper URDF得到8关节/11碰撞mesh；正式验收另存。
已有资产账保留世界米制坐标、实测支撑面与实例ID，适合构造单房视图。

## Options

继续PyBullet；MuJoCo固定臂单房；先选移动平台再跨房。

## Decision

按用户要求，后续机器人验证使用MuJoCo，优先固定Piper单房安装/操作。
PyBullet历史结果保留且不继承为MuJoCo通过。跨房图生成保留，导航/跨房执行待移动平台。
先独立验收MuJoCo模型转换与控制，再独立验收安装位置；不把二者混作抓放成功。

## Rationale

固定基座实际能力与当前硬件一致；安装坐标、支撑与物体布局是抓放的必要前置。

## Consequences

新增SPEC-035后端模型、036单房安装；旧生产Isaac与论文支线不随本决定切换。
MuJoCo mesh采用凸碰撞近似，容器内腔及抓放仍需单独验证。
