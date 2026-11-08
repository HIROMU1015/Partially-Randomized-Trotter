# CDF0/r64対ENTRY E2/r16：独立128/armの公開原本

科学コードcommit `4f73bec3184523207777ce7f09c75c7cdfd3cb31`、SOURCE SHA256 `934bfff3923609b5add4b9ec33ea2a9f2f3ac9004f41d7768f77251e62fb6f89`。

[結果・科学条件・資格](../../../../docs/research/research_c_cdf0_entry_independent_price_results_20261108.md)。Linear H4/1Å/STO-3G、number CDF8/元順LD2、線形Z再配置、partial S2/s1/δ=.1/K0/m0..12、own-ground oracle・状態準備除外、P0/fenced serial。理想law/exact-nativeの条件付きfinite-bit価格proxy。単一名目95% Welch差区間で`NOMINAL_MODEL_ENTRY_LOWER`、実CA/全law/FTは未資格。

- `summary.json`：実行時の固定条件、既存helperが保存した新主比較とarm summary。CIの再計算ではない。
- `native_samples256.csv`：新CDF0/r64の128行と新ENTRY/r16の128行。旧標本を含まない。
- `raw_manifest.json.gz`：science/run01内の全2882ファイルの相対pathとSHA256。
- `json_records.json.gz`：テキスト原本1588件。各valueは元のUTF-8本文で、JSON/JSONLを含む。
- `raw_binary_bundle.tar.gz.part000`〜`part011`：binary原本1294件。QPY、reference/ground NPZ、worker mmap/cache/pickleのbyte原本をtar.gzへ包装し、4 MiBずつに分割した。pickleを実行・decodeしない。
- `../execution/result_audit.json`：全2405 COMPLETE/FAIL0、全256 correctness記録、SOURCE/input保全、archive/part hashと全raw byteの照合。

分割を順に結合したarchiveは49918123 bytes、SHA256 `66ecca06a110c1cd7aff3c7eff7325bade0805e3f788c5cab59792865c01b3ee`。rawは合計1288518354 bytes。元rawは実行worktreeにも保持する。gitには主結果原本と上記の可逆packageを保存した。外部での科学再実行やCI再現を実施したという意味ではない。

このpublicationディレクトリから次の標準Pythonだけでfresh出力へ原本を復元・hash照合できる。配列・回路をdecodeせず、統計・科学計算を開始しない。

```bash
python3 - /tmp/independent-price-raw-20261108 <<'PY'
from pathlib import Path
import sys, json, gzip, tarfile, io, hashlib
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
audit = json.loads(Path('../execution/result_audit.json').read_text())['raw']
parts = []
for entry in audit['parts']:
    data = Path(entry['path']).read_bytes()
    assert len(data) == entry['bytes']
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
    parts.append(data)
archive = b''.join(parts)
assert hashlib.sha256(archive).hexdigest() == audit['archive_sha256']
records = json.loads(gzip.decompress(Path('json_records.json.gz').read_bytes()))['files']
def write(name, data):
    path = (out / name).resolve()
    assert path.is_relative_to(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
for name, text in records.items():
    write(name, text.encode('utf-8'))
with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
    for member in tar:
        assert member.isfile()
        write(member.name, tar.extractfile(member).read())
expected = json.loads(gzip.decompress(Path('raw_manifest.json.gz').read_bytes()))['files']
actual = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in out.rglob('*') if p.is_file()}
assert actual == expected
print('PASS:', len(actual), 'exact raw files restored; no science executed')
PY
```

科学runはretry/resume0、正常mandatory STOP。旧CDF0 final64比較の`UNRESOLVED_PRECISION_TARGET`と旧ENTRY対PR-aware結果は変更しない。
