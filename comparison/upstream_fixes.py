"""Auditable compatibility fixes for exact pinned upstream source files."""
import ast
import hashlib
from pathlib import Path

ORIGINAL_SHA256 = '94ec98e86f4f32320ce7fedc681f6ec740ec13f9c0ebbb88166b882af06852ef'
UNUSED_IMPORT = 'from vggt_visfeat.layers.attention_utils import per_image_confidence_minmem\n'
FIX_ID = 'vggt360-remove-unused-missing-attention-utils-import-v1'

def fix_vggt360(root):
    path=Path(root)/'vggt_visfeat/layers/attention.py'
    raw=path.read_bytes();source=raw.decode('utf-8')
    if UNUSED_IMPORT not in source:
        # Accept only the exact original file with our one-line change.
        lines=source.splitlines(keepends=True)
        lines.insert(17,UNUSED_IMPORT)
        if hashlib.sha256(''.join(lines).encode()).hexdigest()!=ORIGINAL_SHA256:
            raise RuntimeError(f'Unexpected upstream contents: {path}; no changes made')
        return FIX_ID
    if hashlib.sha256(raw).hexdigest()!=ORIGINAL_SHA256:
        raise RuntimeError(f'Unexpected upstream contents: {path}; no changes made')
    tree=ast.parse(source)
    if any(isinstance(n,ast.Name) and n.id=='per_image_confidence_minmem' for n in ast.walk(tree)):
        raise RuntimeError('The imported symbol is used; refusing to remove it')
    patched=source.replace(UNUSED_IMPORT,'',1)
    compile(patched,str(path),'exec')
    path.write_text(patched,encoding='utf-8')
    print(f'Applied upstream compatibility fix: {FIX_ID}',flush=True)
    return FIX_ID
