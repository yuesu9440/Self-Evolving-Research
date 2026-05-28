#!/usr/bin/env python3
"""
三种复合体系模型构建 + 专业 3D 可视化
============================================
体系1: PVA 链团 + MXene 二维片
体系2: PVA 链团 + CMC-CS 纤维
体系3: MXene 二维片 + CMC-CS 纤维

用 ASE 构建原子级模型，Matplotlib 3D 渲染
"""
import numpy as np
import os, sys, math, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 颜色方案 (Nature 期刊风格)
# ============================================================
COLORS = {
    'PVA':   '#2196F3',   # 蓝色
    'MXene': '#9C27B0',   # 紫色
    'CMC':   '#4CAF50',   # 绿色
    'CS':    '#FF9800',   # 橙色
    'water': '#B3E5FC',   # 浅蓝
    'bond':  '#888888',   # 灰色键
}

# ============================================================
# 1. 构建 PVA 链团 (聚乙烯醇)
# ============================================================

def build_pva_chain(n_monomers=30, start_pos=None, direction=None):
    """构建单条 PVA 链 - (CH2-CHOH)n """
    if start_pos is None:
        start_pos = np.array([0.0, 0.0, 0.0])
    if direction is None:
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
    
    atoms_coords = []
    bond_length = 1.54  # C-C 键长
    
    pos = start_pos.copy()
    dir_vec = direction.copy()
    
    for i in range(n_monomers):
        # 主链 C-C (2个碳)
        c1 = pos.copy()
        pos = pos + dir_vec * bond_length
        c2 = pos.copy()
        
        # -OH 方向 (侧基)
        perp = np.cross(dir_vec, np.random.randn(3))
        perp = perp / (np.linalg.norm(perp) + 1e-10)
        oh_pos = c2 + perp * 1.0
        
        atoms_coords.append({'pos': c1, 'elem': 'C', 'type': 'PVA'})
        atoms_coords.append({'pos': c2, 'elem': 'C', 'type': 'PVA'})
        atoms_coords.append({'pos': oh_pos, 'elem': 'O', 'type': 'PVA'})
        
        # 小幅随机转向
        dir_vec = dir_vec + np.random.randn(3) * 0.3
        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-10)
        
        # 增加弯曲
        bend = np.random.uniform(-0.5, 0.5)
        dir_vec = dir_vec + np.cross(dir_vec, np.array([0, 0, 1])) * bend * 0.1
        dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-10)
    
    return atoms_coords

def build_pva_chain_cluster(n_chains=5, chain_length=40, spread=15):
    """构建 PVA 链团簇"""
    all_atoms = []
    
    for _ in range(n_chains):
        start = np.random.uniform(-spread/2, spread/2, 3)
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        chain = build_pva_chain(chain_length, start, direction)
        all_atoms.extend(chain)
    
    # 统计
    n_c = sum(1 for a in all_atoms if a['elem'] == 'C')
    n_o = sum(1 for a in all_atoms if a['elem'] == 'O')
    print(f"  PVA链团: 总原子={len(all_atoms)} (C={n_c}, O={n_o})")
    
    return all_atoms

# ============================================================
# 2. 构建 MXene 片 (复用之前代码)
# ============================================================

def build_mxene_sheet(nx=3, ny=3):
    """构建 Ti₃C₂Tₓ MXene 二维片"""
    a = 3.05
    c = 19.5
    
    struct_frac = {
        'Ti1': (2/3, 1/3, 0.125),
        'Ti2': (1/3, 2/3, 0.500),
        'Ti3': (2/3, 1/3, 0.875),
        'C1':  (1/3, 2/3, 0.250),
        'C2':  (2/3, 1/3, 0.750),
        'O1':  (1/3, 2/3, 0.025),
        'O2':  (2/3, 1/3, 0.975),
    }
    elem_map = {'Ti1':'Ti','Ti2':'Ti','Ti3':'Ti','C1':'C','C2':'C','O1':'O','O2':'O'}
    
    atoms = []
    sqrt3 = np.sqrt(3)
    
    for ix in range(nx):
        for iy in range(ny):
            for name, (fx, fy, fz) in struct_frac.items():
                x = a * (ix + fx) - a/2 * (iy + fy)
                y = a * sqrt3/2 * (iy + fy)
                z = c * fz
                
                atoms.append({
                    'pos': np.array([x, y, z]),
                    'elem': elem_map[name],
                    'type': 'MXene'
                })
    
    # 居中
    center = np.mean([a['pos'] for a in atoms], axis=0)
    for a in atoms:
        a['pos'] -= center
    
    n_ti = sum(1 for a in atoms if a['elem'] == 'Ti')
    n_c = sum(1 for a in atoms if a['elem'] == 'C')
    n_o = sum(1 for a in atoms if a['elem'] == 'O')
    print(f"  MXene片: 总原子={len(atoms)} (Ti={n_ti}, C={n_c}, O={n_o})")
    
    return atoms

# ============================================================
# 3. 构建 CMC-CS 纤维 (纤维素-壳聚糖复合纤维)
# ============================================================

def build_cmccs_fiber(length=60, radius=4, n_chains=8):
    """
    构建 CMC-CS 复合纤维
    CMC (羧甲基纤维素): 葡萄糖单元 + CH2COOH 侧基
    CS (壳聚糖): 葡萄糖胺单元 + NH2 侧基
    简化为: 多股螺旋扭结的粗粒化纤维
    """
    atoms = []
    
    # 纤维主干方向
    main_dir = np.array([0, 0, 1])
    
    for chain in range(n_chains):
        # 每条链围绕纤维轴螺旋
        theta_offset = 2 * np.pi * chain / n_chains
        radius_chain = radius * (0.5 + 0.2 * np.random.random())
        
        for step in range(length):
            z = step * 1.2 - length * 0.6
            theta = theta_offset + step * 0.15  # 螺旋
            
            x = radius_chain * np.cos(theta)
            y = radius_chain * np.sin(theta)
            
            # 添加轻微波动
            x += np.random.uniform(-0.3, 0.3)
            y += np.random.uniform(-0.3, 0.3)
            
            pos = np.array([x, y, z])
            
            # 每3个原子一个 O (CMC 特征) 或 N (CS 特征)
            if chain < n_chains // 2:
                elem = 'O' if step % 3 == 0 else 'C'
            else:
                elem = 'N' if step % 4 == 0 else 'C'
            
            atoms.append({
                'pos': pos,
                'elem': elem,
                'type': 'CMC-CS'
            })
    
    n_c = sum(1 for a in atoms if a['elem'] == 'C')
    n_o = sum(1 for a in atoms if a['elem'] == 'O')
    n_n = sum(1 for a in atoms if a['elem'] == 'N')
    print(f"  CMC-CS纤维: 总原子={len(atoms)} (C={n_c}, O={n_o}, N={n_n})")
    
    return atoms

# ============================================================
# 4. 复合体系组装
# ============================================================

def build_system1_pva_mxene():
    """体系1: PVA链团 + MXene二维片"""
    print("\n" + "="*60)
    print("  体系1: PVA 链团 + MXene 二维片")
    print("="*60)
    
    mxene = build_mxene_sheet(nx=6, ny=5)
    pva = build_pva_chain_cluster(n_chains=4, chain_length=30, spread=12)
    
    # 将 MXene 放在 x-y 平面，PVA 在 z 正方向一侧
    for a in mxene:
        a['pos'][2] -= 5
    
    for a in pva:
        a['pos'][2] += 8
    
    all_atoms = mxene + pva
    
    return all_atoms, {'MXene': mxene, 'PVA': pva}

def build_system2_pva_cmccs():
    """体系2: PVA链团 + CMC-CS纤维"""
    print("\n" + "="*60)
    print("  体系2: PVA 链团 + CMC-CS 纤维")
    print("="*60)
    
    fiber = build_cmccs_fiber(length=50, radius=3, n_chains=6)
    pva = build_pva_chain_cluster(n_chains=4, chain_length=25, spread=10)
    
    # 纤维竖直放置，PVA 围绕在周围
    for a in fiber:
        a['pos'] = a['pos'] * 1.0
    
    for a in pva:
        a['pos'][0] += 10  # 放在纤维右侧
        a['pos'][2] *= 0.8
    
    all_atoms = fiber + pva
    
    return all_atoms, {'CMC-CS': fiber, 'PVA': pva}

def build_system3_mxene_cmccs():
    """体系3: MXene + CMC-CS纤维"""
    print("\n" + "="*60)
    print("  体系3: MXene 二维片 + CMC-CS 纤维")
    print("="*60)
    
    mxene = build_mxene_sheet(nx=5, ny=4)
    fiber = build_cmccs_fiber(length=40, radius=2.5, n_chains=6)
    
    # MXene 水平放置，纤维竖直穿过
    for a in mxene:
        a['pos'][2] -= 5
    
    for a in fiber:
        a['pos'][0] += 8  # 放在 MXene 上方
        a['pos'][2] *= 2.0
        a['pos'][1] *= 0.5
    
    all_atoms = mxene + fiber
    
    return all_atoms, {'MXene': mxene, 'CMC-CS': fiber}

# ============================================================
# 5. 专业 3D 可视化
# ============================================================

def plot_composite_system(all_atoms, components, title, filename,
                           view_angle=(25, -55)):
    """
    绘制复合体系的 3D 结构图
    风格: 球棍模型 + 半透明表面 + 光照效果
    """
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # 从各组分提取坐标和元素
    colors_map = {
        'PVA':    {'C': ('#2196F3', 60), 'O': ('#F44336', 70)},
        'MXene':  {'Ti': ('#9C27B0', 80), 'C': ('#424242', 50), 'O': ('#F44336', 70)},
        'CMC-CS': {'C': ('#4CAF50', 50), 'O': ('#F44336', 60), 'N': ('#FF9800', 65)},
    }
    
    # 绘制每个组分
    comp_labels = []
    for comp_name, comp_atoms in components.items():
        for elem_data in colors_map.get(comp_name, {}).items():
            elem, (color, size) = elem_data
            subset = [a for a in comp_atoms if a['elem'] == elem]
            if not subset:
                continue
            xs = [a['pos'][0] for a in subset]
            ys = [a['pos'][1] for a in subset]
            zs = [a['pos'][2] for a in subset]
            
            ax.scatter(xs, ys, zs, c=color, s=size,
                      alpha=0.8, edgecolors='white', linewidth=0.3,
                      zorder=3)
        
        comp_labels.append(comp_name)
    
    # 绘制键连接 (简化: 距离近的原子间画线)
    for comp_name, comp_atoms in components.items():
        if len(comp_atoms) > 300:
            continue  # 原子太多时跳过键
        positions = np.array([a['pos'] for a in comp_atoms])
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if 0.8 < dist < 2.0:
                    ax.plot([positions[i][0], positions[j][0]],
                           [positions[i][1], positions[j][1]],
                           [positions[i][2], positions[j][2]],
                           color='#999999', linewidth=0.5, alpha=0.3, zorder=1)
    
    # 绘制半透明连接/界面区域 (装饰性)
    all_pos = np.array([a['pos'] for a in all_atoms])
    center = all_pos.mean(axis=0)
    spread = all_pos.std(axis=0).max()
    
    # 绘制半透明背景球 (营造立体感)
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    r = spread * 1.2
    
    # 一个半透明的环境球
    xs = r * np.outer(np.cos(u), np.sin(v)) + center[0]
    ys = r * np.outer(np.sin(u), np.sin(v)) + center[1]
    zs = r * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_wireframe(xs, ys, zs, color='#E0E0E0', alpha=0.08, linewidth=0.2)
    
    # 标注组分
    legend_elements = []
    legend_colors = {
        'PVA': '#2196F3',
        'MXene': '#9C27B0',
        'CMC-CS': '#4CAF50',
    }
    for cname in comp_labels:
        color = legend_colors.get(cname, '#888888')
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=color, markersize=12, label=cname)
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=13,
             framealpha=0.9, edgecolor='#CCCCCC')
    
    # 设置视角
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # 轴标签
    ax.set_xlabel('X (Å)', fontsize=11, labelpad=8)
    ax.set_ylabel('Y (Å)', fontsize=11, labelpad=8)
    ax.set_zlabel('Z (Å)', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    # 等比例轴
    max_range = max(
        all_pos[:,0].max() - all_pos[:,0].min(),
        all_pos[:,1].max() - all_pos[:,1].min(),
        all_pos[:,2].max() - all_pos[:,2].min()
    ) / 2
    mid = all_pos.mean(axis=0)
    for i, axis in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        axis((mid[i] - max_range*1.1, mid[i] + max_range*1.1))
    
    # 网格和背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#E0E0E0')
    ax.yaxis.pane.set_edgecolor('#E0E0E0')
    ax.zaxis.pane.set_edgecolor('#E0E0E0')
    ax.grid(True, alpha=0.15)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [保存] {save_path}")
    return save_path

# ============================================================
# 6. 俯视图 + 渲染增强
# ============================================================

def plot_top_view(all_atoms, components, title, filename):
    """俯视图 (z轴投影)"""
    fig, ax = plt.subplots(figsize=(10, 9))
    
    cmap = {
        'PVA':   {'color': '#2196F3', 'size': 15},
        'MXene': {'color': '#9C27B0', 'size': 20},
        'CMC-CS':{'color': '#4CAF50', 'size': 12},
    }
    
    for comp_name, comp_atoms in components.items():
        style = cmap.get(comp_name, {'color': '#888888', 'size': 10})
        xs = [a['pos'][0] for a in comp_atoms]
        ys = [a['pos'][1] for a in comp_atoms]
        ax.scatter(xs, ys, c=style['color'], s=style['size'],
                  alpha=0.5, edgecolors='white', linewidth=0.2,
                  label=comp_name)
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_title(f'{title} - 俯视图', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename.replace('.png', '_top.png'))
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {save_path}")
    return save_path

# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 70)
    print("  三种复合体系模型构建 + 3D 可视化")
    print("=" * 70)
    
    systems = [
        ('system1_pva_mxene', '体系1: PVA 链团 + MXene 二维片', 
         build_system1_pva_mxene, (20, -60)),
        ('system2_pva_cmccs', '体系2: PVA 链团 + CMC-CS 纤维', 
         build_system2_pva_cmccs, (25, -45)),
        ('system3_mxene_cmccs', '体系3: MXene 二维片 + CMC-CS 纤维', 
         build_system3_mxene_cmccs, (20, -55)),
    ]
    
    for prefix, title, builder, view in systems:
        print(f"\n{'='*60}")
        print(f"  构建: {title}")
        print(f"{'='*60}")
        
        all_atoms, components = builder()
        
        # 统计
        total = len(all_atoms)
        print(f"\n  体系总原子: {total}")
        for cname, catoms in components.items():
            print(f"    {cname}: {len(catoms)} 个原子")
        
        # 3D 图
        plot_composite_system(all_atoms, components, title,
                             f'{prefix}.png', view)
        
        # 俯视图
        plot_top_view(all_atoms, components, title, f'{prefix}.png')
        
        # 保存结构数据 (xyz格式可用OVITO打开)
        xyz_path = os.path.join(OUTPUT_DIR, f'{prefix}.xyz')
        with open(xyz_path, 'w') as f:
            f.write(f"{total}\n{title}\n")
            for a in all_atoms:
                f.write(f"{a['elem']} {a['pos'][0]:.4f} {a['pos'][1]:.4f} {a['pos'][2]:.4f}\n")
        print(f"  [保存] {xyz_path}")
    
    print(f"\n{'='*70}")
    print("  ✅ 全部完成！共生成 3 个体系 × 3 个文件 = 9 个文件")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
