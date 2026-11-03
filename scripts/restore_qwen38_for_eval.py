"""Restore the existing Qwen3.8-27B archive to an isolated local directory."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

SOURCE = Path('/home/tiger/on-policy-distillation-with-visual-primitives/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
MANIFEST = Path('/home/tiger/cross-image-visual-primitives/paper_revision/revision_artifacts/disk_cleanup_20261022T200516Z/qwen38_archive/archive_manifest.json')
DEST = Path(__file__).resolve().parents[1] / '.local_models/qwen38-27b'


def restore(row):
    dest = DEST / row['name']
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + '.partial')
        if tmp.exists():
            raise RuntimeError(f'Previous partial file exists: {tmp}')
        subprocess.run(['/opt/tiger/arnold/hdfs_client/hdfs', 'dfs', '-get', row['restore_from'], str(tmp)], check=True, timeout=1800)
        with tmp.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if tmp.stat().st_size != row['bytes'] or digest != row['sha256']:
            raise RuntimeError(f'Checksum mismatch: {tmp}')
        tmp.rename(dest)
    else:
        with dest.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if dest.stat().st_size != row['bytes'] or digest != row['sha256']:
            raise RuntimeError(f'Existing file does not match archive: {dest}')
    print(f"verified {row['name']}", flush=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for source in SOURCE.iterdir():
        if source.is_file() and not source.name.endswith('.hdfs.json'):
            target = DEST / source.name
            if not target.exists():
                shutil.copy2(source, target)
    rows = json.loads(MANIFEST.read_text())['verified_shards']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(restore, rows))
    print('RESTORE COMPLETE', DEST, flush=True)


if __name__ == '__main__':
    main()
