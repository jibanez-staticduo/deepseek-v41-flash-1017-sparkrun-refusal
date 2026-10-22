#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Prepare a COPY-only image context from a locally built, pinned upstream image."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TREES = {
    'vllm': '4d6851c8bb1397a55254ba88905a87d9785b98dc',
    'b12x': '1a647a311ca55d45070ab83d60668d9c8fb8886c',
    'nccl': '6af10aa730a64fcee8781d62340a9c9184e79469',
    'tilelang': '222b895d04ee6cd75179155d06e0c0d450636f51',
    'tile_kernels': 'cb2fa41c8af290204c81c547ed678ebf957cca53',
    'sparknet': 'f72599fa212e894d67ee114553e628ab05daf2da',
}


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_image(info):
    if info['Architecture'] != 'arm64':
        raise ValueError('base must be an ARM64 image')
    labels = info['Config'].get('Labels') or {}
    for component, tree in EXPECTED_TREES.items():
        if labels.get('local.spark3.' + component + '.tree') != tree:
            raise ValueError('base source tree differs for ' + component)
    return info['Id']


def prepare(base_image, output, directions):
    manifest = json.loads((ROOT / 'patches/manifest.json').read_text())
    info = json.loads(run('docker', 'image', 'inspect', base_image))[0]
    image_id = verify_image(info)
    # Docker's FROM parses bare sha256 IDs as repository names. Give the
    # verified local image a content-named tag, then check it resolves exactly.
    build_base = 'local/deepseek-refusal-base:' + image_id.removeprefix('sha256:')
    run('docker', 'image', 'tag', image_id, build_base)
    if json.loads(run('docker', 'image', 'inspect', build_base))[0]['Id'] != image_id:
        raise ValueError('local build base tag does not resolve to the verified image')
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    source = output / 'overlay'
    source.mkdir()
    container = run('docker', 'create', '--platform', 'linux/arm64', image_id, 'true')
    try:
        for relative, hashes in manifest['files'].items():
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            run('docker', 'cp', container + ':/opt/spark3/candidate/vllm/' + relative, str(destination))
            if digest(destination) != hashes['before']:
                raise ValueError('base source checksum mismatch: ' + relative)
    finally:
        run('docker', 'rm', container)
    patch_path = ROOT / 'patches/0001-request-scoped-refusal.patch'
    subprocess.run(['patch', '--batch', '--forward', '-p1', '-i', str(patch_path)], cwd=source, check=True)
    for relative, hashes in manifest['files'].items():
        if digest(source / relative) != hashes['after']:
            raise ValueError('patched source checksum mismatch: ' + relative)
        path = source / relative
        text = path.read_text()
        path.write_text('# Modified by Staticduo contributors, 2026-10-22: request-scoped refusal integration.\n' + text)
    if digest(directions) != 'd9820fff33a289ed9b0ce8a08669512250bae35d13fba8598acb521a25a2daaa':
        raise ValueError('directions do not match the tested candidate')
    shutil.copytree(ROOT / 'payload', output / 'payload', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (output / 'scripts').mkdir()
    shutil.copy2(ROOT / 'scripts/launch.py', output / 'scripts/launch.py')
    (output / 'assets').mkdir()
    shutil.copy2(directions, output / 'assets/refusal_dirs_1017.safetensors')
    for name in ['Dockerfile', '.dockerignore', 'runtime.json']:
        shutil.copy2(ROOT / name, output / name)
    notices = output / 'notices'
    notices.mkdir()
    for name in ['LICENSE', 'NOTICE', 'ATTRIBUTION.md']:
        shutil.copy2(ROOT / name, notices / name)
    shutil.copytree(ROOT / 'licenses', notices / 'licenses')
    receipt = {'base_image_id': image_id, 'build_base': build_base,
               'upstream_recipe_revision': manifest['upstream_recipe_revision'],
               'source_files': manifest['files'],
               'modified_files': {name: digest(source / name) for name in manifest['files']},
               'patch_sha256': digest(patch_path),
               'integration_files': {str(path.relative_to(output)): digest(path)
                                     for path in sorted((output / 'payload').glob('*.py'))},
               'launch_sha256': digest(output / 'scripts/launch.py'),
               'runtime_sha256': digest(ROOT / 'runtime.json'),
               'directions_sha256': digest(directions)}
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'context': str(output), 'base_image_id': image_id, 'build_base': build_base,
                      'build': f'docker build --platform linux/arm64 --build-arg BASE_IMAGE={build_base} -t YOUR_IMAGE {output}'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-image', required=True)
    parser.add_argument('--output', type=Path, default=Path('build/context'))
    parser.add_argument('--directions', type=Path, default=ROOT / 'assets/refusal_dirs_1017.safetensors')
    args = parser.parse_args()
    prepare(args.base_image, args.output, args.directions)


if __name__ == '__main__':
    main()
