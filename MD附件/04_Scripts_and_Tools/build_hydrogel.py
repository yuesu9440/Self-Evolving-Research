#!/usr/bin/env python3
"""
水凝胶自动建模脚本 — LAMMPS Python 接口控制
=============================================
功能:
  1. 自动构建聚合物网络（单链、交联结构）
  2. 写入 LAMMPS data 文件
  3. 通过 LAMMPS 执行能量最小化 + 平衡模拟
  4. 提取热力学数据

研究方向: 水凝胶基柔性电子器件
"""
import numpy as np
import os, sys, json
from lammps import lammps

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(OUTPUT_DIR)

# ============================================================
# 配置参数（改这里调整建模参数）
# ============================================================
CONFIG = {
    'n_chains': 10,              # 聚合物链数
    'chain_length': 30,          # 每条链单体数
    'crosslink_density': 0.05,   # 交联密度
    'temperature': 300,          # K
    'pressure': 1.0,             # atm
    'equil_steps': 20000,        # 平衡步数
    'prod_steps': 50000,         # 生产步数
}

# ============================================================
# 1. 构建聚合物链（随机行走 + 交联）
# ============================================================

def build_polymer(n_chains, chain_length, crosslink_density, bead_dist=1.5, box_size=50):
    """用随机行走构建交联聚合物网络"""
    print(f"\n{'='*60}")
    print(f"  [建链] 构建 {n_chains} 条链 × {chain_length} 单体")
    print(f"  [交联] 密度: {crosslink_density:.1%}")
    print(f"{'='*60}")
    
    np.random.seed(42)
    half = box_size / 2
    
    atoms = []
    bonds = []
    angles = []
    
    # --- 建链 ---
    for chain in range(n_chains):
        # 随机起始点
        pos = np.random.uniform(-half + 5, half - 5, 3)
        
        for m in range(chain_length):
            aid = len(atoms) + 1
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            step = bead_dist * np.array([
                np.sin(phi)*np.cos(theta),
                np.sin(phi)*np.sin(theta),
                np.cos(phi)
            ])
            pos = pos + step
            # 约束在盒子内
            pos = np.clip(pos, -half + 2, half - 2)
            
            atoms.append({
                'id': aid, 'mol': chain+1, 'type': 1,
                'charge': 0.0,
                'x': pos[0], 'y': pos[1], 'z': pos[2]
            })
            
            # 链内键
            if m > 0:
                bonds.append({
                    'id': len(bonds)+1, 'type': 1,
                    'a1': aid-1, 'a2': aid
                })
            # 链内角
            if m > 1:
                angles.append({
                    'id': len(angles)+1, 'type': 1,
                    'a1': aid-2, 'a2': aid-1, 'a3': aid
                })
    
    # --- 交联 ---
    n_cross = int(len(atoms) * crosslink_density)
    positions = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    mol_ids = np.array([a['mol'] for a in atoms])
    existing = set((b['a1'], b['a2']) for b in bonds)
    _ = existing.union((b['a2'], b['a1']) for b in bonds)
    
    crosslinks = []
    np.random.seed(123)
    attempts = 0
    
    while len(crosslinks) < n_cross and attempts < n_cross * 100:
        attempts += 1
        i, j = np.random.randint(0, len(atoms), 2)
        if mol_ids[i] == mol_ids[j]: continue
        if (i+1, j+1) in existing: continue
        
        dist = np.linalg.norm(positions[i] - positions[j])
        if dist < 4.0:
            crosslinks.append({
                'id': len(bonds) + len(crosslinks) + 1,
                'type': 2, 'a1': i+1, 'a2': j+1
            })
            existing.add((i+1, j+1))
    
    bonds += crosslinks
    print(f"  原子: {len(atoms)}, 键: {len(bonds)}, 交联: {len(crosslinks)}")
    return atoms, bonds, angles

# ============================================================
# 2. 写入 LAMMPS data 文件
# ============================================================

def write_data(atoms, bonds, angles, filename='hydrogel.data'):
    """写入 LAMMPS 可读的 data 文件"""
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    margin = 5
    xlo, xhi = coords[:,0].min() - margin, coords[:,0].max() + margin
    ylo, yhi = coords[:,1].min() - margin, coords[:,1].max() + margin
    zlo, zhi = coords[:,2].min() - margin, coords[:,2].max() + margin
    
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as f:
        f.write(f"Hydrogel model\n\n")
        f.write(f"{len(atoms)} atoms\n{len(bonds)} bonds\n{len(angles)} angles\n\n")
        f.write(f"2 atom types\n2 bond types\n1 angle types\n\n")
        f.write(f"{xlo:.3f} {xhi:.3f} xlo xhi\n")
        f.write(f"{ylo:.3f} {yhi:.3f} ylo yhi\n")
        f.write(f"{zlo:.3f} {zhi:.3f} zlo zhi\n\n")
        f.write("Atoms\n\n")
        for a in atoms:
            f.write(f"{a['id']} {a['mol']} {a['type']} {a['charge']} {a['x']:.6f} {a['y']:.6f} {a['z']:.6f}\n")
        f.write("\nBonds\n\n")
        for b in bonds:
            f.write(f"{b['id']} {b['type']} {b['a1']} {b['a2']}\n")
        f.write("\nAngles\n\n")
        for a in angles:
            f.write(f"{a['id']} {a['type']} {a['a1']} {a['a2']} {a['a3']}\n")
    
    print(f"  [写入] {filename}  盒子: [{xlo:.1f}, {xhi:.1f}]")
    return path

# ============================================================
# 3. LAMMPS 模拟流程
# ============================================================

def run_lammps_simulation(data_file):
    """执行完整的 LAMMPS 模拟（minimize → NPT 平衡 → 生产）"""
    print(f"\n{'='*60}")
    print("  [LAMMPS] 启动模拟引擎")
    print(f"{'='*60}")
    
    lmp = lammps()
    
    # --- 基础设置 ---
    lmp.command("units real")
    lmp.command("atom_style full")
    lmp.command("boundary p p p")
    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify delay 5 every 1")
    
    # 读取结构
    lmp.command(f"read_data {data_file}")
    
    # 质量 + 力场参数（粗粒化通用参数）
    lmp.command("mass 1 15.0")   # 单体质量 (CH2 ~14, 稍大一点)
    lmp.command("mass 2 15.0")   # 交联点质量
    lmp.command("pair_style lj/cut 12.0")
    lmp.command("pair_coeff * * 0.2 3.8")
    lmp.command("bond_style harmonic")
    lmp.command("bond_coeff 1 100.0 1.5")
    lmp.command("bond_coeff 2 50.0 2.0")
    lmp.command("angle_style harmonic")
    lmp.command("angle_coeff 1 50.0 109.5")
    lmp.command("special_bonds lj 0.0 0.0 0.5")
    
    # 输出设置
    lmp.command("thermo 500")
    lmp.command("thermo_style custom step temp pe ke press vol")
    
    T = CONFIG['temperature']
    P = CONFIG['pressure']
    
    # --- 1. 能量最小化 ---
    print("\n  [1/4] 能量最小化...")
    lmp.command("minimize 1.0e-4 1.0e-6 1000 10000")
    print("    最小化完成")
    
    # --- 2. NVT 升温 ---
    print("  [2/4] NVT 升温...")
    lmp.command(f"velocity all create {T} 12345")
    lmp.command(f"fix 1 all nvt temp {T} {T} 100.0")
    lmp.command("run 10000")
    lmp.command("unfix 1")
    
    # --- 3. NPT 平衡 ---
    print(f"  [3/4] NPT 平衡 ({CONFIG['equil_steps']} 步)...")
    lmp.command(f"fix 2 all npt temp {T} {T} 100.0 iso {P} {P} 1000.0")
    lmp.command(f"dump 1 all custom 5000 equil.dump id type x y z")
    lmp.command(f"run {CONFIG['equil_steps']}")
    lmp.command("undump 1")
    lmp.command("unfix 2")
    
    # --- 4. 生产模拟 ---
    print(f"  [4/4] 生产模拟 ({CONFIG['prod_steps']} 步)...")
    lmp.command(f"fix 3 all npt temp {T} {T} 100.0 iso {P} {P} 1000.0")
    lmp.command(f"dump 2 all custom 10000 prod.dump id type x y z vx vy vz")
    lmp.command(f"run {CONFIG['prod_steps']}")
    lmp.command("undump 2")
    lmp.command("unfix 3")
    
    # --- 提取结果 ---
    print("  [提取] 热力学数据 (查看 log.lammps)...")
    print(f"    生产模拟已完成，请检查 log.lammps 获取能量/温度/压力数据")
    print(f"    轨迹文件: equil.dump, prod.dump")
    
    lmp.close()
    print("  [完成] LAMMPS 模拟结束 ✅")

# ============================================================
# 4. 主程序
# ============================================================

def main():
    print("=" * 70)
    print("  LAMMPS 水凝胶自动建模系统")
    print(f"  链数={CONFIG['n_chains']}, 链长={CONFIG['chain_length']}")
    print(f"  交联密度={CONFIG['crosslink_density']:.0%}")
    print("=" * 70)
    
    # Step 1: 构建结构
    atoms, bonds, angles = build_polymer(
        CONFIG['n_chains'], CONFIG['chain_length'], CONFIG['crosslink_density']
    )
    
    # Step 2: 写入 data 文件
    data_file = write_data(atoms, bonds, angles, 'hydrogel.data')
    
    # Step 3: 运行模拟
    run_lammps_simulation(data_file)
    
    print(f"\n{'='*70}")
    print("  ✅ 全部完成！输出文件:")
    for f in ['hydrogel.data', 'equil.dump', 'prod.dump', 'log.lammps']:
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(fp):
            print(f"    {f}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
