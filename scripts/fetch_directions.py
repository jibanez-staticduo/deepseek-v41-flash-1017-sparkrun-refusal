#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Staticduo contributors.
"""Fetch a pinned author artifact and extract raw ZIP storage; never unpickle."""
import argparse
import hashlib
import json
import struct
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

REPO = 'pqhaz/DeepSeek-V4.1-Flash-1017-heretic'
REVISION = '12d58ad0878b5008e80d720251adb969a0069412'
SOURCE_SHA = '87e81bffab36b046786eb7a3888c8d12c69ddef145235e363e0d4fb2c8fe31b4'
PICKLE_SHA = '2bb142115470f795d985a6c1f51a9b7423281a569d5dc9aeb957830ddeca7b99'
STORAGE_SHA = '98d65378f2167fed790b92ffb8c6b9aa2ac866b128126b0efc7803e10108ea13'
OUTPUT_SHA = 'd9820fff33a289ed9b0ce8a08669512250bae35d13fba8598acb521a25a2daaa'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(source, output):
    data = Path(source).read_bytes()
    if len(data) > 8 * 1024 * 1024 or sha(data) != SOURCE_SHA:
        raise ValueError('source artifact does not match the reviewed revision')
    with zipfile.ZipFile(source) as archive:
        if archive.read('directions/byteorder') != b'little':
            raise ValueError('unexpected byte order')
        # This exact pickle metadata was inspected separately for shape/stride/storage.
        # Checking its digest permits raw extraction without evaluating its instructions.
        if sha(archive.read('directions/data.pkl')) != PICKLE_SHA:
            raise ValueError('unreviewed tensor metadata')
        raw = archive.read('directions/data/3')
        if sha(raw) != STORAGE_SHA or len(raw) != 40 * 5120 * 4:
            raise ValueError('projected storage does not match reviewed data')
    vectors = np.frombuffer(raw, dtype='<f4').reshape(40, 5120)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError('invalid refusal directions')
    header = {'__metadata__': {
        'model': 'deepseek-ai/DeepSeek-V4.1-Flash-1017',
        'revision': '1bae0e1088993b9b82de08f59de5be5a34e42170',
        'source': REPO + '@' + REVISION,
        'scope': 'backbone attention WO-B layers2..39; MTP/DSpark untouched',
        'status': 'candidate numerical analysis only; no runtime efficacy validated',
    }}
    chunks = []
    offset = 0
    for layer in range(2, 40):
        chunk = vectors[layer].tobytes()
        header[f'layers.{layer}.attn.wo_b'] = {'dtype': 'F32', 'shape': [5120],
                                            'data_offsets': [offset, offset + len(chunk)]}
        offset += len(chunk)
        chunks.append(chunk)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    result = struct.pack('<Q', len(encoded)) + encoded + b''.join(chunks)
    if sha(result) != OUTPUT_SHA:
        raise ValueError('generated directions do not match the tested artifact')
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != result:
        raise ValueError('refusing to replace a different artifact')
    destination.write_bytes(result)
    return {'source_repo': REPO, 'revision': REVISION, 'source_sha256': SOURCE_SHA,
            'sha256': OUTPUT_SHA, 'bytes': len(result), 'layers': [2, 39], 'hidden': 5120}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='already downloaded pinned ablation_directions.pt')
    parser.add_argument('--output', type=Path, default=Path('assets/refusal_dirs_1017.safetensors'))
    parser.add_argument('--download-dir', type=Path, default=Path('build/downloads'))
    args = parser.parse_args()
    source = args.source
    if source is None:
        args.download_dir.mkdir(parents=True, exist_ok=True)
        source = args.download_dir / 'ablation_directions.pt'
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/ablation_directions.pt'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        if sha(data) != SOURCE_SHA:
            raise ValueError('download checksum mismatch')
        source.write_bytes(data)
    print(json.dumps(extract(source, args.output), indent=2))


if __name__ == '__main__':
    main()
