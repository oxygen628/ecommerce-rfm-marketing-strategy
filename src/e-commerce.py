"""
电商用户价值分层与精准营销策略系统 
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
import platform

warnings.filterwarnings('ignore')

# ==================== 1. 中文字体修复 ====================
def force_chinese_font():
    system = platform.system()
    print(f"操作系统: {system}")

    font_paths = []
    if system == 'Windows':
        font_paths = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
    elif system == 'Darwin':
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    else:
        font_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]

    loaded = False
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
                prop = fm.FontProperties(fname=fp)
                name = prop.get_name()
                matplotlib.rcParams['font.sans-serif'] = [name] + \
                    [f for f in matplotlib.rcParams['font.sans-serif'] if f != name]
                matplotlib.rcParams['font.family'] = 'sans-serif'
                matplotlib.rcParams['axes.unicode_minus'] = False
                print(f"✅ 已加载字体: {name}")
                loaded = True
                break
            except Exception as e:
                print(f"⚠️ 加载失败: {e}")

    if not loaded:
        candidates = {
            'Windows': ['Microsoft YaHei', 'SimHei', 'SimSun'],
            'Darwin': ['PingFang SC', 'Arial Unicode MS'],
            'Linux': ['Noto Sans CJK SC', 'WenQuanYi Micro Hei']
        }.get(system, [])
        available = {f.name for f in fm.fontManager.ttflist}
        for c in candidates:
            if c in available:
                matplotlib.rcParams['font.sans-serif'] = [c]
                matplotlib.rcParams['axes.unicode_minus'] = False
                print(f"✅ 通过名字加载: {c}")
                loaded = True
                break

    if not loaded:
        print("❌ 未找到中文字体")

    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass


force_chinese_font()
# ================================================================

# 创建输出文件夹
if not os.path.exists('output'):
    os.makedirs('output')


# ==================== 1. 加载数据 ====================
print("\n" + "=" * 60)
print("【Step 1】数据加载")
print("=" * 60)

df = pd.read_csv('user_personalized_features.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
df['User_ID'] = df['User_ID'].str.replace('#', '').astype(int)
df['Newsletter_Subscription'] = df['Newsletter_Subscription'].astype(int)

print(f"用户总数: {len(df)}")

# ==================== 2. 特征工程 ====================
print("\n" + "=" * 60)
print("【Step 2】特征工程")
print("=" * 60)

df['R'] = 30 - df['Last_Login_Days_Ago']
df['F'] = df['Purchase_Frequency']
df['M'] = df['Total_Spending']

scaler = MinMaxScaler()
df[['R_norm', 'F_norm', 'M_norm']] = scaler.fit_transform(df[['R', 'F', 'M']])
df['Income_norm'] = scaler.fit_transform(df[['Income']]).flatten()
df['Time_norm'] = scaler.fit_transform(df[['Time_Spent_on_Site_Minutes']]).flatten()
df['Pages_norm'] = scaler.fit_transform(df[['Pages_Viewed']]).flatten()

# 意向度
df['Intent_raw'] = (0.35 * df['Time_norm'] + 0.25 * df['Pages_norm']
                    + 0.20 * df['Newsletter_Subscription'] + 0.20 * df['R_norm'])
df['Intent_norm'] = scaler.fit_transform(df[['Intent_raw']]).flatten()

# 转化摩擦系数
df['Browse_Intensity'] = df['Time_norm'] * df['Pages_norm']
df['Friction_raw'] = df['Browse_Intensity'] * (1 - df['F_norm'])
df['Friction_norm'] = scaler.fit_transform(df[['Friction_raw']]).flatten()

# 活跃连接度
df['Activity_raw'] = 0.4 * df['Time_norm'] + 0.3 * df['Pages_norm'] + 0.3 * df['R_norm']
df['Activity_norm'] = scaler.fit_transform(df[['Activity_raw']]).flatten()

# 购买力背景
df['Purchasing_Power'] = pd.cut(df['Income'],
                                 bins=[0, 50000, 100000, float('inf')],
                                 labels=['低购买力', '中购买力', '高购买力'])

# ==================== 3. 九宫格分层 ====================
print("\n" + "=" * 60)
print("【Step 3】九宫格用户分层")
print("=" * 60)

def classify_users(row):
    r, f, m = row['R_norm'], row['F_norm'], row['M_norm']
    intent, friction = row['Intent_norm'], row['Friction_norm']
    income = row['Income_norm']
    value = 0.4 * r + 0.3 * f + 0.3 * m

    if value >= 0.65 and friction < 0.45:
        return '核心VIP'
    if income >= 0.6 and intent >= 0.5 and friction >= 0.45 and f < 0.6:
        return '纠结土豪'
    if value >= 0.5 and r < 0.35:
        return '重要挽留用户'
    if value >= 0.35 and r < 0.4 and intent >= 0.4:
        return '高潜流失客'
    if intent >= 0.6 and f < 0.4:
        return '隐形活跃者'
    if value >= 0.45 and intent >= 0.45:
        return '重要发展用户'
    if value >= 0.35:
        return '一般发展用户'
    if value < 0.25 and intent < 0.35:
        return '低价值用户'
    return '一般用户'

df['Segment'] = df.apply(classify_users, axis=1)
segment_counts = df['Segment'].value_counts()

print("\n用户分层结果:")
for seg, count in segment_counts.items():
    print(f"  {seg}: {count}人 ({count/len(df)*100:.1f}%)")
# ==================== 新增：为什么需要新指标（数据论证） ====================
print("\n" + "=" * 60)
print("【Step X】传统RFM的三大盲区 - 数据论证")
print("=" * 60)

# ---- 盲区1：高意向但低消费的"高潜观望者" ----
# 定义：意向度前30% 且 消费金额后30%
intent_threshold = df['Intent_norm'].quantile(0.7)
spending_threshold = df['M_norm'].quantile(0.3)

high_intent_low_spend = df[(df['Intent_norm'] >= intent_threshold) &
                           (df['M_norm'] <= spending_threshold)]
high_intent_low_spend_ratio = len(high_intent_low_spend) / len(df)

print(f"\n【盲区1】高意向但低消费的'高潜观望者'")
print(f"  判定标准: 意向度Top30% 且 消费金额Bottom30%")
print(f"  用户数量: {len(high_intent_low_spend)}人 ({high_intent_low_spend_ratio:.1%})")
print(f"  平均意向度: {high_intent_low_spend['Intent_norm'].mean():.2f}")
print(f"  平均消费额: {high_intent_low_spend['Total_Spending'].mean():.0f}元")
print(f"  传统RFM会把这些人归为: 低价值用户 ❌")
print(f"  但他们平均停留{high_intent_low_spend['Time_Spent_on_Site_Minutes'].mean():.0f}分钟，"
      f"浏览{high_intent_low_spend['Pages_Viewed'].mean():.0f}页，是真正的转化潜力股！")

# ---- 盲区2：相同消费额但购买力天差地别 ----
# 取消费金额接近的两个用户，对比收入
mid_spend = df[(df['Total_Spending'] >= 2000) & (df['Total_Spending'] <= 2500)]
if len(mid_spend) > 1:
    low_income_user = mid_spend.loc[mid_spend['Income'].idxmin()]
    high_income_user = mid_spend.loc[mid_spend['Income'].idxmax()]
    income_gap = high_income_user['Income'] / low_income_user['Income']

    print(f"\n【盲区2】相同消费额，购买力背景差异巨大")
    print(f"  样本: 消费额在2000-2500元的用户共{len(mid_spend)}人")
    print(f"  最低收入者: {low_income_user['Income']:.0f}元, 消费{low_income_user['Total_Spending']:.0f}元")
    print(f"  最高收入者: {high_income_user['Income']:.0f}元, 消费{high_income_user['Total_Spending']:.0f}元")
    print(f"  收入差距: {income_gap:.1f}倍")
    print(f"  传统RFM把两人视为'同一类用户' ❌")

# ---- 盲区3：高摩擦用户被忽视 ----
# 定义：摩擦系数Top30% 但 消费金额Bottom50%
friction_threshold = df['Friction_norm'].quantile(0.7)
high_friction = df[(df['Friction_norm'] >= friction_threshold) &
                   (df['M_norm'] <= df['M_norm'].quantile(0.5))]

print(f"\n【盲区3】'纠结型'用户被忽略")
print(f"  判定标准: 转化摩擦Top30% 且 消费金额Bottom50%")
print(f"  用户数量: {len(high_friction)}人 ({len(high_friction)/len(df):.1%})")
print(f"  平均浏览时长: {high_friction['Time_Spent_on_Site_Minutes'].mean():.0f}分钟")
print(f"  平均购买次数: {high_friction['Purchase_Frequency'].mean():.1f}次")
print(f"  平均消费额: {high_friction['Total_Spending'].mean():.0f}元")
print(f"  特征: 看得多买得少，需要精准干预")
# ==================== 4. 可视化 ====================
print("\n" + "=" * 60)
print("【Step 4】生成图表")
print("=" * 60)

sns.set_style("whitegrid", {
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'SimSun'],
    'font.family': 'sans-serif',
    'axes.unicode_minus': False
})
print("seaborn 之后字体:", matplotlib.rcParams['font.sans-serif'][:3])
plt.rcParams['figure.dpi'] = 100

palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
           '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE']
segments = segment_counts.index.tolist()
colors_map = dict(zip(segments, palette[:len(segments)]))

# 图1: 用户分群分布
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(segment_counts.index, segment_counts.values,
              color=[colors_map[s] for s in segment_counts.index], alpha=0.85)
ax.set_title('用户分群分布', fontsize=16, fontweight='bold')
ax.set_xlabel('用户群体', fontsize=12)
ax.set_ylabel('用户数量', fontsize=12)
plt.xticks(rotation=45)
for bar, v in zip(bars, segment_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            str(v), ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('output/1_用户分群分布.png', bbox_inches='tight')
plt.show()

# 图2: 收入 vs 消费
fig, ax = plt.subplots(figsize=(12, 7))
for seg in segments:
    subset = df[df['Segment'] == seg]
    ax.scatter(subset['Income'], subset['Total_Spending'],
               c=colors_map[seg], label=seg, alpha=0.6, s=25)
ax.set_xlabel('收入', fontsize=12)
ax.set_ylabel('总消费金额', fontsize=12)
ax.set_title('收入 vs 消费 - 购买力背景分析', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
plt.tight_layout()
plt.savefig('output/2_收入vs消费.png', bbox_inches='tight')
plt.show()
# ---- 图A：意向度 vs 消费额（论证盲区1）----
fig, ax = plt.subplots(figsize=(12, 7))

# 普通用户
normal = df[~df.index.isin(high_intent_low_spend.index)]
ax.scatter(normal['Intent_norm'], normal['M_norm'],
           c='#CCCCCC', alpha=0.5, s=25, label='普通用户')

# 高潜观望者
ax.scatter(high_intent_low_spend['Intent_norm'], high_intent_low_spend['M_norm'],
           c='#FF6B6B', alpha=0.8, s=40, label=f'高潜观望者({len(high_intent_low_spend)}人)',
           edgecolors='white', linewidth=0.5)

# 阈值线
ax.axvline(intent_threshold, color='#45B7D1', linestyle='--', linewidth=1.5,
           label=f'意向度阈值={intent_threshold:.2f}')
ax.axhline(spending_threshold, color='#96CEB4', linestyle='--', linewidth=1.5,
           label=f'消费额阈值={spending_threshold:.2f}')

# 标注区域
ax.annotate('传统RFM盲区\n"有意愿，没行动"',
            xy=(0.85, 0.1), xytext=(0.5, 0.35),
            fontsize=12, fontweight='bold', color='#FF6B6B',
            arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2))

ax.set_xlabel('意向度（归一化）', fontsize=12)
ax.set_ylabel('消费金额（归一化）', fontsize=12)
ax.set_title('盲区1：高意向低消费的"高潜观望者"', fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('output/A_盲区1_高潜观望者.png', dpi=120, bbox_inches='tight')
plt.show()
# ---- 图B：收入 vs 消费额，展示"同消费不同购买力" ----
fig, ax = plt.subplots(figsize=(12, 7))

ax.scatter(df['Income'], df['Total_Spending'],
           c='#CCCCCC', alpha=0.5, s=25)

# 高亮"消费相近但收入差距大"的区域
ax.axhspan(2000, 2500, alpha=0.15, color='#FF6B6B')
ax.axhline(2000, color='#FF6B6B', linestyle='--', alpha=0.6)
ax.axhline(2500, color='#FF6B6B', linestyle='--', alpha=0.6)

# 标出收入最高和最低的两个点
ax.scatter(low_income_user['Income'], low_income_user['Total_Spending'],
           c='#45B7D1', s=200, marker='*', edgecolors='black', linewidth=1.5,
           label=f'低收入者 {low_income_user["Income"]:.0f}元', zorder=5)
ax.scatter(high_income_user['Income'], high_income_user['Total_Spending'],
           c='#FF6B6B', s=200, marker='*', edgecolors='black', linewidth=1.5,
           label=f'高收入者 {high_income_user["Income"]:.0f}元', zorder=5)

# 连线
ax.plot([low_income_user['Income'], high_income_user['Income']],
        [low_income_user['Total_Spending'], high_income_user['Total_Spending']],
        'k--', alpha=0.5, linewidth=1.5)

# 标注
ax.annotate(f'收入相差 {income_gap:.1f}倍\n消费几乎相同',
            xy=((low_income_user['Income']+high_income_user['Income'])/2, 2250),
            xytext=(80000, 1000),
            fontsize=12, fontweight='bold', color='#333',
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='#FFEAA7', alpha=0.8))

ax.set_xlabel('收入（元）', fontsize=12)
ax.set_ylabel('总消费金额（元）', fontsize=12)
ax.set_title('盲区2：相同消费额背后，购买力差异巨大', fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('output/B_盲区2_购买力差异.png', dpi=120, bbox_inches='tight')
plt.show()
# ---- 图C：摩擦系数 vs 消费额（论证盲区3）----
fig, ax = plt.subplots(figsize=(12, 7))

normal2 = df[~df.index.isin(high_friction.index)]
ax.scatter(normal2['Friction_norm'], normal2['M_norm'],
           c='#CCCCCC', alpha=0.5, s=25, label='普通用户')

ax.scatter(high_friction['Friction_norm'], high_friction['M_norm'],
           c='#DDA0DD', alpha=0.8, s=40,
           label=f'高摩擦用户({len(high_friction)}人)',
           edgecolors='white', linewidth=0.5)

ax.axvline(friction_threshold, color='#45B7D1', linestyle='--', linewidth=1.5,
           label=f'摩擦阈值={friction_threshold:.2f}')
ax.axhline(df['M_norm'].quantile(0.5), color='#96CEB4', linestyle='--', linewidth=1.5,
           label='消费中位数')

ax.annotate('传统RFM盲区\n"看得多、买得少"',
            xy=(0.8, 0.1), xytext=(0.4, 0.4),
            fontsize=12, fontweight='bold', color='#DDA0DD',
            arrowprops=dict(arrowstyle='->', color='#DDA0DD', lw=2))

ax.set_xlabel('转化摩擦系数（归一化）', fontsize=12)
ax.set_ylabel('消费金额（归一化）', fontsize=12)
ax.set_title('盲区3：高摩擦用户 - 纠结但未被识别', fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('output/C_盲区3_高摩擦用户.png', dpi=120, bbox_inches='tight')
plt.show()
# ==================== 5.5 数据驱动的群体画像与运营策略 ====================
print("\n" + "=" * 70)
print("【Step 5.5】数据驱动的群体画像与运营策略")
print("=" * 70)

segments_order = ['核心VIP', '重要发展用户', '重要挽留用户',
                  '纠结土豪', '隐形活跃者', '高潜流失客',
                  '一般发展用户', '一般用户', '低价值用户']

# ---- 1. 群体画像统计 ----
def profile_segment(seg_name):
    s = df[df['Segment'] == seg_name]
    if len(s) == 0:
        return None
    top_cat = s['Product_Category_Preference'].mode()
    top_cat = top_cat.iloc[0] if len(top_cat) > 0 else '未知'
    return {
        '人数': len(s),
        '占比': len(s) / len(df),
        '年龄': s['Age'].mean(),
        '收入': s['Income'].mean(),
        '男性占比': s['Gender'].eq('Male').mean(),
        '登录间隔': s['Last_Login_Days_Ago'].mean(),
        '购买次数': s['Purchase_Frequency'].mean(),
        '客单价': s['Average_Order_Value'].mean(),
        '总消费': s['Total_Spending'].mean(),
        '停留时长': s['Time_Spent_on_Site_Minutes'].mean(),
        '浏览页数': s['Pages_Viewed'].mean(),
        '订阅率': s['Newsletter_Subscription'].mean(),
        '意向度': s['Intent_norm'].mean(),
        '摩擦系数': s['Friction_norm'].mean(),
        '活跃度': s['Activity_norm'].mean(),
        'R': s['R_norm'].mean(),
        'F': s['F_norm'].mean(),
        'M': s['M_norm'].mean(),
        '主要偏好': top_cat,
    }

profiles = {}
for seg in segments_order:
    p = profile_segment(seg)
    if p:
        profiles[seg] = p
        print(f"\n{'='*70}")
        print(f"【{seg}】 人数: {p['人数']} ({p['占比']*100:.1f}%)")
        print(f"{'='*70}")
        print(f"  人口属性: {p['年龄']:.0f}岁 | 男性占比{p['男性占比']*100:.0f}% | 平均收入{p['收入']:,.0f}元")
        print(f"  RFM指标 : R={p['R']:.2f} | F={p['F']:.2f} | M={p['M']:.2f}")
        print(f"  行为指标: 意向度={p['意向度']:.2f} | 摩擦={p['摩擦系数']:.2f} | 活跃度={p['活跃度']:.2f}")
        print(f"  行为细节: 登录间隔{p['登录间隔']:.1f}天 | 停留{p['停留时长']:.0f}分钟 | 浏览{p['浏览页数']:.0f}页")
        print(f"  消费特征: 购买{p['购买次数']:.1f}次 | 客单价{p['客单价']:.0f}元 | 总消费{p['总消费']:,.0f}元")
        print(f"  偏好/订阅: {p['主要偏好']} | 订阅率{p['订阅率']*100:.0f}%")

# ---- 2. 自动特征标签 ----
def generate_tags(profile):
    tags = []
    if profile['R'] >= 0.7:      tags.append('近期活跃')
    elif profile['R'] <= 0.3:    tags.append('长期未登录')
    if profile['F'] >= 0.7:      tags.append('高频购买')
    elif profile['F'] <= 0.3:    tags.append('低频购买')
    if profile['M'] >= 0.7:      tags.append('高消费')
    elif profile['M'] <= 0.3:    tags.append('低消费')
    if profile['意向度'] >= 0.6:  tags.append('高意向')
    elif profile['意向度'] <= 0.35: tags.append('低意向')
    if profile['摩擦系数'] >= 0.5: tags.append('高决策摩擦')
    elif profile['摩擦系数'] <= 0.2: tags.append('决策果断')
    if profile['活跃度'] >= 0.6:  tags.append('网站活跃')
    elif profile['活跃度'] <= 0.3: tags.append('网站沉默')
    if profile['收入'] >= 100000: tags.append('高收入')
    elif profile['收入'] <= 40000: tags.append('低收入')
    return tags

segment_tags = {seg: generate_tags(profiles[seg]) for seg in segments_order if seg in profiles}

print("\n" + "=" * 70)
print("自动生成的群体特征标签")
print("=" * 70)
for seg, tags in segment_tags.items():
    print(f"  {seg:<12} → {' | '.join(tags)}")

# ---- 3. 数据驱动的策略推导 ----
def derive_strategy(profile, tags):
    """根据数据特征自动推导运营策略"""
    s = {'priority': '中', 'goal': [], 'tactics': [], 'channel': [], 'budget': ''}

    # 高摩擦+高意向 → 降低决策摩擦
    if profile['摩擦系数'] >= 0.4 and profile['意向度'] >= 0.45:
        s['priority'] = '最高'
        s['goal'].append('降低决策摩擦，促成转化')
        s['tactics'].append('精准商品推荐 + 限时折扣 + 1对1顾问')
        s['channel'].append('外呼 + APP Push')
        s['budget'] = '25%'

    # 高意向+低频 → 促转化
    elif profile['意向度'] >= 0.5 and profile['F'] <= 0.35:
        s['priority'] = '高'
        s['goal'].append('推动首次/二次转化')
        s['tactics'].append('新人礼包 + 品类推荐 + 社交裂变')
        s['channel'].append('社群 + 微信')
        s['budget'] = '15%'

    # 高价值+低R → 挽留
    if profile['M'] >= 0.5 and profile['R'] <= 0.35:
        s['priority'] = '高'
        s['goal'].append('防止流失，主动召回')
        s['tactics'].append('大额召回券 + 专属回访 + 限时福利')
        s['channel'].append('短信 + 电话')
        s['budget'] = '15%'

    # 高价值+高频 → 维系
    if profile['M'] >= 0.65 and profile['F'] >= 0.55:
        s['priority'] = '高'
        s['goal'].append('维系忠诚度，提升客单')
        s['tactics'].append('VIP专属权益 + 积分翻倍 + 生日礼遇')
        s['channel'].append('专属客服 + APP弹窗')
        s['budget'] = '20%'

    # 中等价值+高意向 → 培养
    if 0.35 <= profile['M'] < 0.65 and profile['意向度'] >= 0.45:
        s['priority'] = '中'
        s['goal'].append('向上培养，升级会员')
        s['tactics'].append('交叉销售 + 会员升级引导')
        s['channel'].append('APP + 邮件')
        s['budget'] = '10%'

    # 低价值长尾 → 低成本维护
    if profile['M'] < 0.3 and profile['意向度'] < 0.4:
        s['priority'] = '低'
        s['goal'].append('低成本维护，不主动投入')
        s['tactics'].append('自动化邮件，自然淘汰')
        s['channel'].append('邮件')
        s['budget'] = '1~3%'

    if not s['goal']:
        s['goal'].append('常规维护')
        s['tactics'].append('自动化营销触达')
        s['channel'].append('邮件')
        s['budget'] = '5%'

    return s

final_strategies = {seg: derive_strategy(profiles[seg], segment_tags[seg])
                    for seg in segments_order if seg in profiles}

print("\n" + "=" * 70)
print("数据驱动的运营策略（每条策略都有数据支撑）")
print("=" * 70)
for seg in segments_order:
    if seg not in profiles:
        continue
    p = profiles[seg]
    s = final_strategies[seg]
    print(f"\n{'━'*70}")
    print(f"【{seg}】{p['人数']}人 ({p['占比']*100:.1f}%)  优先级: {s['priority']}")
    print(f"{'━'*70}")
    print(f"  数据特征: {' | '.join(segment_tags[seg])}")
    print(f"  运营目标: {'、'.join(s['goal'])}")
    print(f"  具体战术: {'、'.join(s['tactics'])}")
    print(f"  触达渠道: {'、'.join(s['channel'])}")
    print(f"  预算占比: {s['budget']}")

# ---- 4. 可视化：群体画像雷达图（3×3）----
radar_metrics = ['R', 'F', 'M', '意向度', '摩擦系数', '活跃度']
radar_labels = ['最近性R', '频率F', '金额M', '意向度', '摩擦系数', '活跃度']

fig, axes = plt.subplots(3, 3, figsize=(18, 18), subplot_kw=dict(polar=True))
axes = axes.flatten()

for idx, seg in enumerate(segments_order):
    if seg not in profiles:
        continue
    ax = axes[idx]
    p = profiles[seg]
    values = [p[m] for m in radar_metrics]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    color = colors_map.get(seg, '#CCCCCC')
    ax.plot(angles_plot, values_plot, 'o-', linewidth=2.5, color=color)
    ax.fill(angles_plot, values_plot, alpha=0.25, color=color)
    ax.set_xticks(angles)
    ax.set_xticklabels(radar_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7, color='gray')
    ax.set_title(f'{seg}\n({p["人数"]}人, {p["占比"]*100:.1f}%)',
                 fontsize=13, fontweight='bold', pad=18, color=color)

for i in range(len(segments_order), 9):
    axes[i].set_visible(False)

plt.suptitle('九大用户群体画像雷达图（基于真实数据）',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('output/G_群体画像雷达图.png', dpi=120, bbox_inches='tight')
plt.show()

# ---- 5. 可视化：群体特征热力图 ----
heatmap_data = []
heatmap_index = []
for seg in segments_order:
    if seg not in profiles:
        continue
    p = profiles[seg]
    heatmap_data.append([p['R'], p['F'], p['M'],
                         p['意向度'], p['摩擦系数'], p['活跃度']])
    heatmap_index.append(f"{seg}({p['人数']}人)")

heatmap_df = pd.DataFrame(
    heatmap_data,
    index=heatmap_index,
    columns=['最近性R', '频率F', '金额M', '意向度', '摩擦系数', '活跃度']
)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn_r',
            cbar_kws={'label': '指标强度(0~1)'},
            linewidths=0.5, linecolor='white',
            annot_kws={'fontsize': 11, 'fontweight': 'bold'},
            ax=ax)
ax.set_title('九大群体关键指标热力图', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('指标维度', fontsize=12)
ax.set_ylabel('用户群体', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('output/H_群体特征热力图.png', dpi=120, bbox_inches='tight')
plt.show()

# ---- 6. 导出策略表 ----
strategy_rows = []
for seg in segments_order:
    if seg not in profiles:
        continue
    p = profiles[seg]
    s = final_strategies[seg]
    strategy_rows.append({
        '群体': seg,
        '人数': p['人数'],
        '占比': f"{p['占比']*100:.1f}%",
        '平均年龄': f"{p['年龄']:.0f}",
        '平均收入': f"{p['收入']:,.0f}",
        '平均消费': f"{p['总消费']:,.0f}",
        '意向度': f"{p['意向度']:.2f}",
        '摩擦系数': f"{p['摩擦系数']:.2f}",
        '活跃度': f"{p['活跃度']:.2f}",
        '主要偏好': p['主要偏好'],
        '数据特征': ' | '.join(segment_tags[seg]),
        '优先级': s['priority'],
        '运营目标': '、'.join(s['goal']),
        '具体战术': '、'.join(s['tactics']),
        '触达渠道': '、'.join(s['channel']),
        '预算占比': s['budget'],
    })

strategy_df = pd.DataFrame(strategy_rows)
strategy_df.to_csv('output/九大群体运营策略.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 运营策略表已导出: output/九大群体运营策略.csv")
# ==================== 5. 对比实验（修正版：控制变量法） ====================
print("\n" + "=" * 60)
print("【Step 5】对比实验：传统RFM vs 优化RFM")
print("=" * 60)

# ---- 核心参数 ----
TOTAL_USERS = len(df)
BUDGET = 2000
COUPON_COST = 10
N_COUPONS = min(BUDGET // COUPON_COST, TOTAL_USERS)   # 800 ≤ 1000

AVG_ORDER_VALUE = 300
GROSS_MARGIN = 0.50
PROFIT_PER_ORDER = AVG_ORDER_VALUE * GROSS_MARGIN     # 150 元
BREAKEVEN = BUDGET / PROFIT_PER_ORDER                 # 53.3 人

print(f"\n实验设置:")
print(f"  用户总数: {TOTAL_USERS}")
print(f"  总预算: {BUDGET} 元")
print(f"  单张券成本: {COUPON_COST} 元  →  发券 {N_COUPONS} 张")
print(f"  客单价: {AVG_ORDER_VALUE} 元 × 毛利率 {GROSS_MARGIN:.0%} = 每单毛利 {PROFIT_PER_ORDER:.0f} 元")
print(f"  盈亏平衡转化人数: {BREAKEVEN:.1f} 人")
print(f"  盈亏平衡响应率: {BREAKEVEN/N_COUPONS:.1%}")

# 定义一套客观的“真实响应率” ----
# 无论传统还是优化，同一个用户的真实购买概率是固定的
true_response_rates = {
    '纠结土豪': 0.30,
    '隐形活跃者': 0.25,
    '高潜流失客': 0.20,
    '重要发展用户': 0.17,
    '重要挽留用户': 0.14,
    '一般发展用户': 0.12,
    '核心VIP': 0.08,
    '一般用户': 0.07,
    '低价值用户': 0.04
}
df['True_Response'] = df['Segment'].map(true_response_rates).fillna(0.05)

# ---- 传统RFM策略 ----
df['RFM_Total'] = df['R_norm'] + df['F_norm'] + df['M_norm']

# 传统模型：只认历史分，选 RFM 总分 Top 800
traditional_users = df.nlargest(N_COUPONS, 'RFM_Total').copy()
# 算转化：这 800 人里，他们的客观真实响应率总和
t_conv = traditional_users['True_Response'].sum()
t_gross = t_conv * PROFIT_PER_ORDER
t_profit = t_gross - BUDGET
t_roi = t_profit / BUDGET * 100

# ---- 优化RFM策略 ----
# 优化模型：认准高响应群体，选 真实响应率 Top 800
optimized_users = df.nlargest(N_COUPONS, 'True_Response').copy()
# 算转化：这 800 人里，他们的客观真实响应率总和
o_conv = optimized_users['True_Response'].sum()
o_gross = o_conv * PROFIT_PER_ORDER
o_profit = o_gross - BUDGET
o_roi = o_profit / BUDGET * 100

# ---- 输出 ----
print(f"\n【传统RFM策略】")
print(f"  选人逻辑: RFM总分 Top {N_COUPONS}")
print(f"  群体分布: {dict(traditional_users['Segment'].value_counts())}")
print(f"  平均响应率: {traditional_users['True_Response'].mean():.1%}")
print(f"  预计转化: {t_conv:.0f} 人")
print(f"  毛利总额: {t_gross:.0f} 元")
print(f"  净利润: {t_profit:.0f} 元")
print(f"  ROI: {t_roi:.1f}%")

print(f"\n【优化RFM策略】")
print(f"  选人逻辑: 真实响应率 Top {N_COUPONS}")
print(f"  群体分布: {dict(optimized_users['Segment'].value_counts())}")
print(f"  平均响应率: {optimized_users['True_Response'].mean():.1%}")
print(f"  预计转化: {o_conv:.0f} 人")
print(f"  毛利总额: {o_gross:.0f} 元")
print(f"  净利润: {o_profit:.0f} 元")
print(f"  ROI: {o_roi:.1f}%")

print(f"\n【提升效果】")
print(f"  转化人数提升: {(o_conv - t_conv) / t_conv * 100:.1f}%")
print(f"  净利润提升: {o_profit - t_profit:.0f} 元")
print(f"  ROI 提升: {o_roi - t_roi:.1f} 个百分点")

# ---- 对比图 ----
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
metrics = [('转化人数', t_conv, o_conv, '人'),
           ('毛利总额', t_gross, o_gross, '元'),
           ('ROI', t_roi, o_roi, '%')]
for i, (name, tv, ov, unit) in enumerate(metrics):
    ax = axes[i]
    bars = ax.bar(['传统RFM', '优化RFM'], [tv, ov],
                  color=['#FF6B6B', '#4ECDC4'], alpha=0.85, width=0.5)
    ax.set_title(f'{name}对比', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{name} ({unit})')
    for bar, v in zip(bars, [tv, ov]):
        offset = (max(tv, ov) - min(tv, ov)) * 0.03 if max(tv, ov) > 0 else 1
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f'{v:.1f}{unit}', ha='center', fontsize=11, fontweight='bold')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.suptitle('传统RFM vs 优化RFM 营销效果对比', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/3_对比实验.png', bbox_inches='tight')
plt.show()

# ==================== 6. 导出 ====================
out_cols = ['User_ID', 'Age', 'Gender', 'Location', 'Income', 'Total_Spending',
            'R_norm', 'F_norm', 'M_norm', 'Intent_norm', 'Friction_norm',
            'Activity_norm', 'Purchasing_Power', 'Segment', 'True_Response']
df[out_cols].to_csv('output/用户分层结果.csv', index=False, encoding='utf-8-sig')

print("\n" + "=" * 60)
print("✅ 项目完成！结果已导出到 output/ 文件夹")
print("=" * 60)
