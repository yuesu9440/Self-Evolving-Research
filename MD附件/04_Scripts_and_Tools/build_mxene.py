#!/usr/bin/env python3
"""
MXene (Ti₃C₂Tₓ) 模型构建 + 3D 可视化
=====================================
结构: Ti₃C₂Tₓ (三层 Ti, 两层 C, 表面-O 终止)
晶系: 六方晶系 (P6₃/mmc)
"""
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(OUTPUT_DIR)

# ============================================================
# MXene 参数
# ============================================================
# Ti₃C₂Tₓ 晶格常数 (Å)
A = 3.05   # a 轴
C = 19.5   # c 轴 (含表面终止层)

# 原子坐标 (分数坐标, 相对于六方晶胞)
# 六方晶胞基本向量:
# a1 = (a, 0, 0)
# a2 = (-a/2, a*sqrt(3)/2, 0)
# a3 = (0, 0, c)

# Ti₃C₂ 原子位置 (分数坐标)
# 参考: Ti₃C₂ - 空间群 P6₃/mmc (No. 194)
STRUCTURE = {
    'Ti1': (0.6667, 0.3333, 0.125),   # 外层 Ti (上)
    'Ti2': (0.3333, 0.6667, 0.500),   # 中间 Ti
    'Ti3': (0.6667, 0.3333, 0.875),   # 外层 Ti (下)
    'C1':  (0.3333, 0.6667, 0.250),   # 上层 C
    'C2':  (0.6667, 0.3333, 0.750),   # 下层 C
    'O1':  (0.3333, 0.6667, 0.025),   # 表面 -O (上)
    'O2':  (0.6667, 0.3333, 0.975),   # 表面 -O (下)
}

# 原子可视化参数
ATOM_STYLES = {
    'Ti': {'color': '#808080', 'size': 80,  'label': 'Ti'},
    'C':  {'color': '#333333', 'size': 50,  'label': 'C'},
    'O':  {'color': '#FF0000', 'size': 60,  'label': 'O'},
}

# ============================================================
# 1. 坐标转换: 分数坐标 → 笛卡尔坐标
# ============================================================

def frac_to_cart(frac_coords, a, c):
    """六方晶系分数坐标转笛卡尔坐标"""
    x_frac, y_frac, z_frac = frac_coords
    sqrt3 = np.sqrt(3)
    x = a * x_frac - a/2 * y_frac
    y = a * sqrt3/2 * y_frac
    z = c * z_frac
    return np.array([x, y, z])

# ============================================================
# 2. 构建超胞
# ============================================================

def build_supercell(nx=3, ny=3, nz=1):
    """构建 MXene 超胞"""
    print("=" * 60)
    print("  MXene Ti₃C₂Tₓ 模型构建")
    print(f"  超胞: {nx}×{ny}×{nz}")
    print("=" * 60)
    
    atoms = []
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                offset = np.array([ix, iy, iz])
                
                for name, frac in STRUCTURE.items():
                    total_frac = (np.array(frac) + offset) / np.array([nx, ny, nz])
                    cart = frac_to_cart(total_frac, A * nx, C * nz)
                    
                    # 确定元素
                    if 'Ti' in name:
                        elem = 'Ti'
                    elif 'C' in name:
                        elem = 'C'
                    else:
                        elem = 'O'
                    
                    atoms.append({
                        'name': name,
                        'element': elem,
                        'x': cart[0],
                        'y': cart[1],
                        'z': cart[2]
                    })
    
    print(f"\n  原子总数: {len(atoms)}")
    for elem in ['Ti', 'C', 'O']:
        count = sum(1 for a in atoms if a['element'] == elem)
        print(f"    {elem}: {count}")
    
    return atoms

# ============================================================
# 3. 写入 LAMMPS data 文件
# ============================================================

def write_lammps_data(atoms, a, c, nx, ny, nz, filename='mxene.data'):
    """写入 LAMMPS data 文件"""
    path = os.path.join(OUTPUT_DIR, filename)
    
    xs = [a['x'] for a in atoms]
    ys = [a['y'] for a in atoms]
    zs = [a['z'] for a in atoms]
    
    xlo, xhi = min(xs) - 1, max(xs) + 1
    ylo, yhi = min(ys) - 1, max(ys) + 1
    zlo, zhi = min(zs) - 1, max(zs) + 1
    
    elem_types = {'Ti': 1, 'C': 2, 'O': 3}
    elem_mass = {'Ti': 47.87, 'C': 12.01, 'O': 16.00}
    
    with open(path, 'w') as f:
        f.write(f"MXene Ti3C2Tx supercell {nx}x{ny}x{nz}\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write("3 atom types\n\n")
        f.write(f"{xlo:.3f} {xhi:.3f} xlo xhi\n")
        f.write(f"{ylo:.3f} {yhi:.3f} ylo yhi\n")
        f.write(f"{zlo:.3f} {zhi:.3f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for elem, tid in elem_types.items():
            f.write(f"{tid} {elem_mass[elem]}  # {elem}\n")
        f.write("\nAtoms\n\n")
        for i, a in enumerate(atoms, 1):
            tid = elem_types[a['element']]
            f.write(f"{i} 1 {tid} 0.0 {a['x']:.6f} {a['y']:.6f} {a['z']:.6f}\n")
    
    print(f"\n  [写入] {path}")
    return path

# ============================================================
# 4. 3D 可视化
# ============================================================

def visualize(atoms, nx=3, ny=3, nz=1, filename='mxene_structure.png'):
    """3D 可视化 MXene 结构"""
    print("\n  [绘图] 生成 3D 结构图...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 按元素分组绘制
    for elem in ['Ti', 'C', 'O']:
        subset = [a for a in atoms if a['element'] == elem]
        xs = [a['x'] for a in subset]
        ys = [a['y'] for a in subset]
        zs = [a['z'] for a in subset]
        
        style = ATOM_STYLES[elem]
        ax.scatter(xs, ys, zs, 
                  c=style['color'], 
                  s=style['size'],
                  label=style['label'],
                  alpha=0.85,
                  edgecolors='white',
                  linewidth=0.5)
    
    # 绘制 Ti-C 键连（同一列上的最近邻）
    print("  [绘图] 绘制化学键...")
    plot_bonds(ax, atoms)
    
    # 标注超胞边界
    x_max = max(a['x'] for a in atoms)
    y_max = max(a['y'] for a in atoms)
    z_max = max(a['z'] for a in atoms)
    x_min = min(a['x'] for a in atoms)
    y_min = min(a['y'] for a in atoms)
    z_min = min(a['z'] for a in atoms)
    
    # 绘制盒子
    for (x1, y1, z1), (x2, y2, z2) in [
        ((x_min, y_min, z_min), (x_max, y_min, z_min)),
        ((x_min, y_min, z_min), (x_min, y_max, z_min)),
        ((x_min, y_min, z_min), (x_min, y_min, z_max)),
        ((x_max, y_min, z_min), (x_max, y_max, z_min)),
        ((x_max, y_min, z_min), (x_max, y_min, z_max)),
        ((x_min, y_max, z_min), (x_max, y_max, z_min)),
        ((x_min, y_max, z_min), (x_min, y_max, z_max)),
        ((x_min, y_min, z_max), (x_max, y_min, z_max)),
        ((x_min, y_min, z_max), (x_min, y_max, z_max)),
        ((x_max, y_max, z_min), (x_max, y_max, z_max)),
        ((x_max, y_min, z_max), (x_max, y_max, z_max)),
        ((x_min, y_max, z_max), (x_max, y_max, z_max)),
    ]:
        ax.plot([x1, x2], [y1, y2], [z1, z2], 'gray', linewidth=0.5, alpha=0.4)
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    ax.set_title(f'MXene Ti₃C₂Tₓ  ({nx}×{ny}×{nz} supercell)', fontsize=14, fontweight='bold')
    
    # 图例
    legend = ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # 设置视角 - 倾斜俯视以展示层状结构
    ax.view_init(elev=20, azim=-60)
    
    # 等比例轴
    max_range = max(x_max-x_min, y_max-y_min, z_max-z_min) / 2
    mid_x, mid_y, mid_z = (x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  [保存] {save_path}")
    
    # 也保存一个俯视图
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for elem in ['Ti', 'C', 'O']:
        subset = [a for a in atoms if a['element'] == elem]
        xs = [a['x'] for a in subset]
        ys = [a['y'] for a in subset]
        style = ATOM_STYLES[elem]
        ax2.scatter(xs, ys, c=style['color'], s=style['size']*0.8,
                   label=style['label'], alpha=0.85, edgecolors='white', linewidth=0.5)
    ax2.set_xlabel('X (Å)', fontsize=12)
    ax2.set_ylabel('Y (Å)', fontsize=12)
    ax2.set_title('MXene Ti₃C₂Tₓ — 俯视图 (top view)', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=11)
    plt.tight_layout()
    top_path = os.path.join(OUTPUT_DIR, filename.replace('.png', '_top.png'))
    plt.savefig(top_path, dpi=200, bbox_inches='tight')
    print(f"  [保存] {top_path}")
    
    plt.close('all')
    print("  [绘图完成] ✅")

def plot_bonds(ax, atoms):
    """绘制 Ti-C 键"""
    for a1 in atoms:
        if a1['element'] != 'Ti':
            continue
        for a2 in atoms:
            if a2['element'] != 'C':
                continue
            dx = a1['x'] - a2['x']
            dy = a1['y'] - a2['y']
            dz = a1['z'] - a2['z']
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if 1.5 < dist < 2.5:  # Ti-C 键长范围
                ax.plot([a1['x'], a2['x']], [a1['y'], a2['y']], [a1['z'], a2['z']],
                       'gray', linewidth=0.8, alpha=0.3)

# ============================================================
# 5. 写入 LAMMPS 输入脚本
# ============================================================

def write_lammps_in(filename='in.mxene'):
    """写入 LAMMPS 输入脚本"""
    content = """# MXene Ti3C2Tx 模拟输入脚本
units           metal
atom_style      atomic
boundary        p p p

# 读取结构
read_data       mxene.data

# 势函数 (EAM 用于 Ti-C, LJ 用于层间)
pair_style      hybrid/overlay eam lj/cut 10.0
pair_coeff      * * eam Ti_C.eam.alloy Ti C
pair_coeff      3 3 lj/cut 0.0067 3.04

# 质量
mass            1 47.867
mass            2 12.011
mass            3 15.999

# 弛豫设置
neighbor        2.0 bin
neigh_modify    delay 10

# 能量最小化
min_style       cg
minimize        1e-8 1e-10 5000 10000

# 输出
compute         mype all pe
compute         mytemp all temp
thermo          100
thermo_style    custom step pe temp vol

# 保存最终结构
write_data      mxene_relaxed.data

print "=== MXene 模型构建完成 ==="
"""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  [写入] {path}")

# ============================================================
# 6. 主流程
# ============================================================

def main():
    # 超胞大小
    NX, NY, NZ = 4, 4, 1
    
    # 构建结构
    atoms = build_supercell(nx=NX, ny=NY, nz=NZ)
    
    # 写入 LAMMPS data
    write_lammps_data(atoms, A, C, NX, NY, NZ)
    
    # 写入 LAMMPS 输入脚本
    write_lammps_in()
    
    # 3D 可视化
    visualize(atoms, nx=NX, ny=NY, nz=NZ)
    
    print(f"\n{'='*60}")
    print("  ✅ MXene 模型构建完成！")
    print(f"  🖼️  结构图: mxene_structure.png")
    print(f"  🖼️  俯视图: mxene_structure_top.png")
    print(f"  📄  data: mxene.data")
    print(f"  📄  in 脚本: in.mxene")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
