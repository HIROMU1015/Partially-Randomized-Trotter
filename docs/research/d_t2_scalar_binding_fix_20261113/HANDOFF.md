# D → C：corrected total scalarのmetadata接続

[REPORT](REPORT.md)／[選択metadata](SCALAR_BINDING_METADATA.json)／[T2の未取得証拠](T2_RERUN_REQUIREMENTS.md)。今回、Dのselectorだけを修正した。C3802bccのexpected_scalar転送・再構成・schedule anchor・resolved scalarのhex gateを再利用し、C branch/code/SOURCEへ変更を加えていない。

## Dの生成仕様

```python
from trottertracks import ld12_price_requests as L

request = L.request_for(root, 'PR-LD2', 12, axes=('X', 'Y'),
                        execution_source=explicit_new_run_or_technical_source)
```

public引数は元と同じ。内部で固定SOURCEの保存receipt/snapshotを照合し、`constant`配列一要素だけを読む。`ham.constant`相当の値とA保存tail identityをPython floatで **c_H+s_R** の順に加算する。H/state復元・prepare、event/path生成、quantum circuit、compileは呼ばない。constant/sourceが取得できないときは例外で停止する。

requestの既存 `physical_context.expected_scalar` は **total**。provenanceは既存`physical_context.input_source`へ次のように追加し、元Dのhash規則でcontext/requestへ束縛する。

```text
input_source.scalar_binding
  hamiltonian_constant: quantity=HAMILTONIAN_CONSTANT, value, source
  tail_identity: quantity=TAIL_EXTRACTED_IDENTITY_COEFFICIENT, value, source
  expected_total: quantity=EXPLICIT_TOTAL_SCALAR, value,
                 rule=BINARY64_HAMILTONIAN_CONSTANT_PLUS_TAIL_IDENTITY_ONCE
```

constant sourceは固定Git locator、snapshot/receiptのpath/SHA/bytes、constant配列のcanonical shape/dtype/SHA、H hash。tail sourceは元A MENU/prefix selectorとtail出典。Cが値を推測したり、Aのtail-only値をtotalとして渡したりしない。

expected_scalarは**検査値**であり、元Runtimeは依然として復元Hとprepareから自身のscalarを計算し、binary64完全一致を確認する。元composeの一回phase適用も変わらない。metadataにc_Hを記載しても追加のphaseを適用しない。

## C3802bccとの最小接続

```python
# 修正済みDから得た同candidateのrequest。Cはtotalを再計算しない。
pc = request['physical_context']
pending = I.price_requests(root, C_request,
    input_source=pc['input_source'],
    execution_source=pc['execution_source'],
    axes=request['axes'], expected_scalar=pc['expected_scalar'])
```

Cの **3802bcc0212325091515325a889fcab20eabd909** がロードするD constructorは748be687版だが、今回`from_b/from_a_job/check_request`のschema・計算式は変更していない。そのため、同configuration/physical/source/axes/preparation_hashならpinned constructorで同じD keyを作れる。新しい`scalar_binding`はinput_sourceの一部としてそのまま通る。**CのAPI追加やconstructor pin更新は不要**で、upstreamで修正済みD selectorを選び、上の既存転送へ正しいsource/totalを渡す差分だけが必要。

今回の人工metadata確認は、Cの固定`_requests/_paths`関数とpinned D748 constructorをGitから読んだ限定境界テスト。C full integrationや科学実行の再検証ではない。正しいtotalのrequestは再構成でき、tail-onlyなresolved scalarはCの既存gateに拒否された。

## ID・旧record・quantityの扱い

新expected totalとsource/provenanceが変わるため新context/request keyを生成する。旧748 coverage・752 T2・240診断のIDやrecordを書き換えず、新requestへ旧path/meanを再ラベルしない。SOURCEが変わった新keyへmean sidecarを付け替えて資格を作らない。

X/Yのphysical contextは共通でも価格応答は軸別。同scheduleの全roundで同じpreparation/source/totalを保つ。今回一件のmetadata artifactは技術tagを持つ未実行requestで、実科学runのIDとしてcopyしない。後続の新SOURCEと選択scopeで再生成する。

B canonical identity、law、q/R/K・signed PF piece、compiler/control/new ZZ/full measured wrapper/P0は不変。CがBへ渡すpath C、S、finite-cohort mean、適格E[C]/区間の区別も不変。現在C/S/D・価格mean/CIは未取得。今回値付きの価格packetやG入力は作らない。

**次：ユーザー**がT2新runの有限scopeを別途承認するか判断。**Codex D**は承認後だけ新SOURCE/出力先で取得し、**C**は必要になった際に既存転送で新requestを受け取る。現在は技術修正の公開照合後 **mandatory STOP**。
