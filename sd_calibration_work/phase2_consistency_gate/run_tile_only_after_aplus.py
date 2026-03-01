#!/usr/bin/env python3
"""
监控A+-only训练完成，然后自动启动tile-only训练
"""
import os
import time
import subprocess
import sys

def main():
    a_plus_task_id = "bf863cf"

    print("等待A+-only训练完成...")
    print(f"Task ID: {a_plus_task_id}")

    # 等待A+-only训练完成
    while True:
        # 检查checkpoint文件是否存在且更新
        ckpt_path = "baseline/runs/mixed_ws4000_sd8960_a_plus_only_hardened/ckpt_last.pth"

        if os.path.exists(ckpt_path):
            # 检查文件的修改时间，确保训练已经完成
            current_time = time.time()
            file_mtime = os.path.getmtime(ckpt_path)

            # 如果文件在最近2分钟内被修改，说明训练刚完成
            if current_time - file_mtime < 120:
                print(f"\nA+-only训练已完成！检测到checkpoint: {ckpt_path}")

                # 等待一段时间确保GPU资源释放
                print("等待10秒以确保GPU资源释放...")
                time.sleep(10)

                # 启动tile-only训练
                print("\n启动SD tile-only训练...")
                tile_cmd = [
                    "python", "sd_calibration_work/phase2_consistency_gate/train_sd_tile_only.py",
                    "--total_steps", "10000",
                    "--batch_size", "16",
                    "--lr", "3e-4",
                    "--log_interval", "200",
                    "--save_interval", "2000"
                ]

                print(f"命令: {' '.join(tile_cmd)}")

                result = subprocess.run(tile_cmd, cwd="/home/yf/soiling_project")
                print(f"\nTile-only训练完成，退出码: {result.returncode}")
                return

        # 每分钟检查一次
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - 等待A+-only训练完成...")
        time.sleep(60)

if __name__ == "__main__":
    main()
