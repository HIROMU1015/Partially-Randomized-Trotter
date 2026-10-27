# Input and SOURCE qualification

2026-10-28 JST。独立検証用read-only資格確認。過去のgrant/one-shot/STOPの解除ではない。

- NPZ：FOUND_HASH_MATCH。`input-0.90.npz` SHA-256 `06672860ed8f9298053b4d84693fbab6d13e38c3166c27dbabcc161f33673c3d`。
- 完全generation-freeze：FOUND_COMPLETE、32配列。file SHA `75d7ddc8dc71ebeec03a6c173397a9b941b492b74e4dc80814d613d83ce56c69`、fingerprint `0fd1de52bce01c351c99efe0535913b2b3fd9a359ee72be36bc75cf847d193e7`。
- 前回の32/32 key・NPY header・C/Fortran格納・canonical byte hash照合を再利用した。今回全array hashをやり直さず、NPZ/freeze実SHAとfingerprint、runtime32 key/shape/dtypeを確認した。
- `allow_pickle=False`、全配列writeable=False。非公開NPZ/array/state/中間行列と個別array hashをGitHubへ出さない。

| SOURCE role | 完全commit |
| --- | --- |
| Original array generator | `049e69919af16ad29a67a217dc7a407d6b1754a6` |
| Dataset/candidate identity source | `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525` |
| Earlier run05 execution | `6d365257770e99022b91d6a38dbee49ee0077503` |
| Later saved candidate execution | `d0726e946f89f8c30a08d383fb8ff48b363629ec` |
| Fixed public donor | `821cc0f6ced57168293ea166644024867b064ffb` |
| Fixed P1 implementation | `2c865c66171fc9d1972b67c1257c57e5684fe26c` |

inputs.py同一blobという事実をsignal/circuits/identity全体の同一版認定へ広げない。旧生成・後続dataset identity・実行SOURCEを分けたまま同じ保存payloadへのbindingを確認した。

| Identity | 完全値 |
| --- | --- |
| input | `7ec5d653bf47f78e3406736db819fcaaad3b046fcc316a533f52462738a5984d` |
| DF | `1601857d7a494e9f2a16b884592712b2c18e674010600168204dcb56c8cf29e7` |
| H | `b7efc24d7b5de66600e6ab73cbb708cc74f0989d8018c3f4fa2f08e86f01b8ed` |
| state | `cef098b502010462335e6d920fb9a0991007114fb9edd490477f3a52c8a3ca29` |
| fixed candidate | `0b2da363d9cc4bd4ddf4a4c3d2cbf10be6a9cf628d7e54b97f3e068af62be888` |
| template | `3b34c4d076c7e64df396acad7c4d70bfab4b5b95ae15e251d5da83e72122d58e` |

候補 `B2-rank3-q1-r4-K4` は旧production_resume02 planに正確に存在し、0.90 Å input record・source/compiler/environment別identityから候補fingerprintを再現した。保存列のrank12、deterministic0/1/2、randomized3..11、元permutationとraw index対応は非公開監査へ保存。新しい順位・候補最適化0。

scalarはnuclearを一回、oneはhspin+保存DF correction。元sector N=4/Nalpha=Nbeta=2/dimension36、full state256、sector state36、bit reverseしたQiskit state、最大振幅pivotの正位相規約を固定した。今回norm/bit/phase/energy/residualだけを既存保存stateで検証し、基底状態を再計算していない。

後続旧公開結果のMAP_COMPLETE_STOP・mandatory_stop=true・next_stage_authorized=falseを読み取った。Aに後続raw runtime/STOP原本が存在すると推測せず、公開固定結果の停止記録と、Aのrun05保全原本を区別した。旧候補signalの投影は32 trajectoryの保存量であり、独立deterministic oracleではない。raw original signal SHAを公開投影に照合したことと、そのraw本文をAで取得したことも区別する。

公開[旧入力provenance](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13/source_input_provenance_v1.json)、[旧固定計画](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/821cc0f6ced57168293ea166644024867b064ffb/artifacts/resource_applicability/track_a_h4_production_resume02/2026-10-11/plan_authorized_v1.json)を実際に取得した。LOCALの2917843…技術検証とfa99868…追加探索は歴史資料として保持し、書換えない。
