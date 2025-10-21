import os
import sys
import shutil
from pathlib import Path

if len(sys.argv) < 2:
    print('Usage: fix_uncond_artifacts.py <run_folder>')
    sys.exit(2)

run_folder = Path(sys.argv[1])
if not run_folder.exists():
    print('Run folder not found:', run_folder)
    sys.exit(1)

uncond = run_folder / 'uncond'
if not uncond.exists() or not uncond.is_dir():
    print('No uncond folder found under', run_folder)
    sys.exit(0)

print('Processing', uncond)

# 1) Keep only the contour_cond_0.png (if present) and rename to contour_uncond.png
contour_candidates = sorted(uncond.glob('contour_cond_*.png'))
if contour_candidates:
    chosen = None
    # Prefer contour_cond_0.png if present
    for p in contour_candidates:
        if p.name == 'contour_cond_0.png':
            chosen = p
            break
    if chosen is None:
        chosen = contour_candidates[0]
    dest = uncond / 'contour_uncond.png'
    try:
        shutil.move(str(chosen), str(dest))
        print('Kept and renamed', chosen.name, '->', dest.name)
    except Exception as e:
        print('Warning: could not move', chosen, e)

    # remove the other contour files
    for p in contour_candidates:
        if p.exists() and p.name != dest.name:
            try:
                p.unlink()
                print('Removed', p.name)
            except Exception as e:
                print('Warning: could not remove', p.name, e)
else:
    print('No contour_cond_*.png files found in uncond')

# 2) Ensure embedding_space.png exists: recreate by loading dataset and plotting φ vs ψ
embed_path = uncond / 'embedding_space.png'
if embed_path.exists():
    print('embedding_space.png already exists in uncond')
else:
    # Try to load dataset using checkpoint config
    ckpt_path = None
    # Look for best_flow.pt in uncond, else in run_folder
    if (uncond / 'best_flow.pt').exists():
        ckpt_path = uncond / 'best_flow.pt'
    elif (run_folder / 'best_flow.pt').exists():
        ckpt_path = run_folder / 'best_flow.pt'

    if ckpt_path is None:
        print('No checkpoint found to infer dataset. Skipping embedding generation.')
    else:
        try:
            import torch
            from fff.data import load_dataset
            from fff.evaluate.tori import convert_to_angles
            import matplotlib.pyplot as plt

            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            config = ckpt.get('config', {})
            ds_name = config.get('dataset', 'torus_protein')
            ds_root = './fff/data'
            if ds_name.startswith('scop'):
                ds_root = '.'
            # load dataset
            print('Loading dataset', ds_name, 'root', ds_root)
            ds = load_dataset(ds_name, root=ds_root)
            trainset, valset = [convert_to_angles(d[:][0]) for d in ds]
            allset = (trainset.cpu(), valset.cpu())
            import torch as _torch
            all_tensor = _torch.cat([trainset, valset], dim=0).cpu()

            fig = plt.figure(figsize=(6,5))
            plt.scatter(all_tensor[:,0].numpy(), all_tensor[:,1].numpy(), s=1, c='black', alpha=0.3)
            plt.xlabel('Phi')
            plt.ylabel('Psi')
            plt.title('Data scatter (phi, psi)')
            plt.tight_layout()
            fig.savefig(str(embed_path), dpi=150)
            plt.close(fig)
            print('Saved embedding_space.png in uncond')
        except Exception as e:
            print('Failed to generate embedding_space.png:', e)

print('Done')
