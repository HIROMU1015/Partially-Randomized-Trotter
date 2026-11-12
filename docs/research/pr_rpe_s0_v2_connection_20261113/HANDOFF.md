# A → B：S0 V2と全round不足証拠

2026-11-13 JST。技術接続のみ。保存実数値の新再解析0、科学call0。

## 固定接続

- 旧S0公開：`7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75`。科学SOURCE：`7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e`。
- B固定SOURCE：`af284d853890155bacd19e61850c196bd1d5c660`（V2基点6a9d7fd36…）。Bの生成/validator/packet/plannerを再利用し、B worktree変更0。
- 新configuration：`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。これはB公開生成関数で固定S0条件を具体化したもの。別のB configurationを受け取る際は完全一致を要求。
- 旧ID：`6cc89360a60b6f5b6554d044193a28f4c92cafa0208e77efb4d1f2ce9f80181c`は元出典として保存。V2へ再ラベルしない。
- H4 DF10/1Å/STO-3G、S2/prefix1/K2/δ=.1/h=.1/M12、S0 exactκ・state/H/snapshot/prior/decoder/law/profile/P0・minkowski/finite_Bを維持。

## Bが読む成果物

| ファイル | 読み方 |
|---|---|
| [V2_CONFIGURATION](metadata01/V2_CONFIGURATION.json) | B canonical whole-schedule identity。physical_candidateとsource_candidate_idは元A ID |
| [V2_N_HANDOFF](metadata01/V2_N_HANDOFF.json) | 元A packet・科学SOURCE保持、m0=64/64・m6=38/38は**nominal**。他11round未評価。全N_X/Y（十分N）・RMSE bound=null |
| [COVERAGE](metadata01/COVERAGE.json) | local mean/ref/error fieldの存在のみ。数値gate・追加nominalは未評価。B plannerの重複実装ではない |
| [B_EVIDENCE_PLAN](metadata01/B_EVIDENCE_PLAN.json) | B本体の26cell、Nは未資格/十分値欠測、G/winner=null。価格入力なし |
| [REPORT](REPORT.md) | 段階A/B/Cの計画、科学的未資格、必要な承認scope |

`N_packet.source`は元S0科学SOURCE、`adapter_source`は今回のmetadata変換SOURCE、`connection_provenance`は旧新IDと元packet locator。元packetを変更せず`A_original_packet`も保持した。controller/plannerのcanonical identityをAで作り直していない。

A module：`src/trottertracks/pr_rpe_rmse_s0_v2.py`。

```python
from trottertracks import pr_rpe_rmse_s0_v2 as A2
# original_configuration/packetは公開7da8a35fの元JSONのみ。
connected = A2.connect(original_configuration, original_packet,
                       root=repo_root, canonical_v2=B_owned_configuration)
```

B既存plannerへは`configuration=connected['configuration']`、`N_packet=connected['N_packet']`を渡す。外側`source_candidate_id`も元physical IDと一致。SOURCE追加拘束を設定する場合、Nの科学SOURCEとmetadata adapterのSOURCEを取り違えない。共通C record API・価格モデル・費用会計の変更はない。PF B0をFULL_PFへ混ぜない。

## 不足と最小実行案

- m0/m6：元S0をcopyするだけ。今回再評価0。
- **m1/R4**：完全一致meanと独立reference pairあり。旧allowance→R-aware式のscalar bridgeが必要。次の最小科学案は、この一候補の保存gate・bias/phase/radius・nominal N_X/Y再解析。**新mean/reference/normalizer/対照0**。固定S0 runnerはm0/m6専用のまま保全し、将来は小さい別scope adapterを使う。旧Hoeffding solverを呼ばない。
- m1用scalar adapterは実carrier 89c91c692…を出典に使う。S0 `numeric_diagnostics`のP1専用locatorをそのまま付けて再利用しない。
- m3/R16・m4/R32：旧rawに一致meanあり。元SOURCE norm gate≤1e−9はあるがobserved exact_tail_norm_errorは欠測のまま。spectral-power backendのallowance/source bridgeを確認し、H4 T=.8/1.6の独立reference pairを補う案（計4 action）。field欠測を0にしない。bridge不十分なら別途承認した同候補mean/対照取得が必要。
- m2,m5,m7–12：完全一致mean欠測、8候補。既存pairを再利用した8 finite mean＋8内部対照が構造上の不足。R255/510/…へ別Rのmeanを代用しない。
- どの局所診断も十分N・全RMSE保証ではない。全roundのfinite-K/state/bias/alias/law/decoder統計証拠、未解決試行loss、精度/信頼度の契約が段階Cに必要。

保存FAIL/NUMERICAL_UNRESOLVED、SOURCE/binding不一致、精度/closure破綻ではSTOP。R/κ/M/候補/閾値変更、科学retryやcache missによる新計算へ自動移行しない。技術テスト47件PASS、実科学0。

**次の担当**：Bはこの未資格packetを接続。ユーザーはm1保存再解析scopeの実行可否を判断。Aは別途明示承認後にだけ実行。段階B/Cはまだ承認されていない。公開照合後mandatory STOP。
