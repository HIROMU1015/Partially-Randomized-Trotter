"""Bind generated T inventory bytes to original freeze receipt and every use."""
import hashlib
import json
from pathlib import Path
from .guard import exclusive_json

INVENTORY = 'synthesis_inventory.json'
RECEIPT = 'synthesis_inventory.freeze.json'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_inventory(out, inventory, provenance, marker_sha256):
    out = Path(out)
    path, receipt = out / INVENTORY, out / RECEIPT
    exclusive_json(path, inventory)
    digest = _sha(path)
    value = {'inventory_path': INVENTORY, 'inventory_sha256': digest,
             'inventory_bytes': path.stat().st_size,
             'unique_keys': len(inventory['unique_keys']),
             'ordered_keys': [row['key'] for row in inventory['unique_keys']],
             'provenance': provenance, 'marker_sha256': marker_sha256,
             'retries': 0, 'exclusive_before_any_synthesis': True}
    exclusive_json(receipt, value)
    path.chmod(0o444); receipt.chmod(0o444)
    # Read-only bits discourage replacement; the original in-memory digests,
    # not owner-writable filesystem permissions, are the integrity authority.
    return {'inventory_sha256': digest, 'inventory_receipt_sha256': _sha(receipt)}


def verify_inventory(out, binding, *, marker_sha256=None, provenance=None):
    out = Path(out)
    raw, receipt_raw = (out / INVENTORY).read_bytes(), (out / RECEIPT).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding['inventory_sha256']:
        raise PermissionError('Original synthesis inventory SHA256 mismatch')
    if hashlib.sha256(receipt_raw).hexdigest() != binding['inventory_receipt_sha256']:
        raise PermissionError('Original inventory freeze receipt SHA256 mismatch')
    receipt, inventory = json.loads(receipt_raw), json.loads(raw)
    if (receipt['inventory_sha256'] != binding['inventory_sha256']
        or receipt['inventory_bytes'] != len(raw) or receipt['inventory_path'] != INVENTORY
        or receipt['ordered_keys'] != [row['key'] for row in inventory['unique_keys']]
        or receipt['unique_keys'] != len(inventory['unique_keys'])):
        raise PermissionError('Inventory freeze record mismatch')
    if marker_sha256 is not None and receipt['marker_sha256'] != marker_sha256:
        raise PermissionError('Inventory marker identity mismatch')
    if provenance is not None and any(receipt['provenance'].get(k) != v for k, v in provenance.items()):
        raise PermissionError('Inventory source/authorization provenance mismatch')
    return inventory


def verify_synthesis_result(row, expected_angle, binding):
    if any(row.get(k) != v for k, v in binding.items()) or row.get('angle') != expected_angle:
        raise PermissionError('Synthesis result does not bind original inventory/key')
