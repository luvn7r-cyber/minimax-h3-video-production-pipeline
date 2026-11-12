# 通用导演分镜与 MiniMax-H3 提示词结构模板

## 一、通用十三段导演分镜模板（Gate 0 分镜拆解阶段）
`人物 → 画风 → 画面景别 → 场景环境 → 核心道具 → 开场状态 → 机位与运镜 → 分秒动作时间轴（含 0.25s 口型起止窗） → 结束状态 → 固定不变条件 → 禁止项 → 台词/歌词 → 点睛音效与环境声`

---

## 二、MiniMax-H3 基础模式三段结构（适用于 `T2VA` / `I2VA` / `FL2VA` / `L2VA`）
必须严格保留英文段落标题与顺序：
```text
integrated_multimodal_description:
[画风、画幅与光影基调] [Shot 1] [景别、人物初始位置与服装外貌正向描述、机位运镜、精确到秒的分段动作与张闭嘴口型变化、收尾定格姿态。Zero text, zero watermarks.]

overall_soundscape:
[与动作毫秒级对齐的环境声、脚步/器物点睛音效、分时段人声演唱或呼吸声]

non_diegetic_music:
[纯器乐配乐情绪与节奏描述；无配乐叙事片填 N/A]
```

---

## 三、MiniMax-H3 全参考模式六段结构（适用于多角色卡 / 多锚点 / 动作参考 `Ref2VA`）
`<Picture N>` 必须与挂载的角色定妆卡顺序严格一一对应：
```text
subject_definitions:
<Subject 1> is [角色A正向外貌、自然肤色、发型与服装细节] from <Picture 1>[，如有参考视频追加：and whose motion choreography and blocking come from <Video 1>].
<Subject 2> is [场景环境描述，锁死灯光布局与透视].

summary:
[keyframe completion 或 reference generation + keyframe completion] [一句话概括本镜时长、核心交互与首尾衔接目标].

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - [明确保留面孔、肤色与服装细节零跳变].
<Subject 2> (appears in [Shot 1]): fully_preserved - [明确保留同一物理场景底板零换景].

detailed_description:
The target video is a [画风与规格描述].
[Shot 1] [完整分秒动作、视线交互、口型开合时间窗与机位运动；显式指明 In <Picture 1> is ...; Zero text, zero logos, zero watermarks.]

overall_soundscape:
[分时段人声与现场环境音效]

non_diegetic_music:
[背景配乐描述或 N/A]
```
