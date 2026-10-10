# G10 one-shot：利用者明示実行authorization receipt

利用者は2026-10-10（JST）に以下の指示を明示した。JSON内の`explicit_execution_instruction`にも同じ文字列を記録する。

```json
"source `05c5ef23fce775a822ab5686f5da2f0d77675864` の固定契約で、authorization-only直接子Aを作成してG10を一回だけ実行してください。retry=0、旧結果・markerは保持し、終了後はmandatory STOPしてください。  "
```

source S：`05c5ef23fce775a822ab5686f5da2f0d77675864`。
contract：`artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json`、SHA256 `71ae2310a08f9d111faab76f27e6de277189ab132370d8996a0e2420c9510818`。
実行branch：`track-b-g10-one-shot-execution-20261010`。
source final reviewは[受領commit 3b490c2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b490c2e8f5f4b71a40bce381b4f32dbe939c39c/docs/research/track_b_G10_source_final_review_20261010.md)に固定。
受領commitを実行HEADに使わず、Sの直接子authorization-only Aを本branchで作る。

Aの変更はauthorization JSONと本receiptだけ。source/contract/target/provider/compiler/precision/capsは変更しない。
Aを公開し、remote SHA=A、clean、critical/protected/runtime一致、fresh result/markerを確認してから
固定runnerを一回だけ呼ぶ。runs=1、retries=0、mandatory_STOP=true。

p=(1/5,3/10,1/2)、x=5/7、同3-system-qubit provider、m3/5/7、17 direct rows/34 axes。
m5は保存native anchorの共通policy再会計のみ。旧G9 source/result/auth/marker/STOPは保持する。
新key上限162/共通cache192、wall1200秒/CPU900秒/RSS512MiB、その他固定capも維持する。

marker消費後はtechnical failure、timeout、numeric guard、serialization failureでも再実行しない。
不完了prefixの勝敗を最終研究判定に使わない。全outcomeでmandatory STOPし、保存値のread-only監査と
証拠公開だけを行う。m9、新p/x/provider、precision/seed/backend、LP/DF/molecule/NPZ/GPU、
実量子shots/trajectory、algorithm採択、次scienceへ自動進行しない。

本receiptはauthorizationを固定する文書であり、G10の完了・有利性・新規性を示す実行結果ではない。
AのSHAはcommit後にpreflight/result/markerへ記録する。自己参照のため本receiptにA自身のSHAを書き込まない。
