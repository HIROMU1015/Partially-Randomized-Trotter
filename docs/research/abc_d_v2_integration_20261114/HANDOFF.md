# B → A/C/D HANDOFF

2026-11-13。先行仕様commitは`575598983059fdf5238da877f62b40250fd9eae9`。[REPORT](REPORT.md)と[INTERFACE](INTERFACE.md)を参照。最終実装は本HANDOFFを含む公開commitを固定して利用する。

## A/C：固定S0

正本はB `030f3f548a35d98a88e62e92a35f132e0243a293`の`docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json`。ID=`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。同ディレクトリのC_REQUESTS.jsonは正式26軸要求。canonical生成・V1/V2 physical対照は既に完了しているので再実行しない。

A75b5c89のV2 packetは正式受入済み。nominal m0/m6のみ64/38、他11未評価、十分N/global bound=null、UNQUALIFIED。C d458b5eは重点2identityと13coverage、Bは26正式requestを供給。Cの平均compiled値は全軸欠測。C1 false/有限cohort、未知不確かさnullを維持する。構造Sや旧HP型Nで補完しない。

再現するmetadata受入CLI（新しい出力path専用、科学実行optionなし）：

```sh
python -B scripts/intake_abc_s0_published.py --output /tmp/abc-s0-published-new.json
```

結果は固定packetへの参照と[PUBLISHED_INTAKE.json](PUBLISHED_INTAKE.json)。元packetを変更・再生成しない。A/Cで新証拠を公開する際は、同canonical identity全体・元physical候補・law・SOURCE・whole schedule/profile/policy、明示cohort/domain/uncertaintyを添付する。

## D：標準4次とκなしlayout

```python
from trottertracks import full_baseline_v2_requests as D

# base/preparation/M/K/q/R等は担当者がSOURCE付きで供給する。
pf_spec = D.full_pf_spec(base, formula='4th', delta_pf=width, M=M, rank=rank)
pf_config = D.generate(pf_spec)                 # κ/Kなし、Full PF

rte_spec = D.full_h_rte_spec(base, M=M, finite_K=K,
                            kappa=None, preparation=preparation)
rte_config = D.generate(rte_spec, layout_by_round=layout_by_round)
packet = D.requests(rte_config)
assert D.validate_requests(rte_config, packet)
```

`formula='4th'`は既存weights('4th')の順序・binary64・負時間を維持し、formula_id=`PF_STANDARD_YOSHIDA4_BINARY64_V1`。m5は別ID。Full/truncated PFは別identity。PRの標準4次は今回非対応。8次は今回新接続・検証しない。

`kappa=None`は新kind `FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1`専用。既存κ辞書を渡せば従来`D_FULL_H_EXPLICIT_PHYSICAL_LAYOUT_V1`を維持する。κを架空生成・逆算しない。明示layoutは`{m:{'q':int,'R_outer':int}}`、mは0..M、欠測roundは未評価。同event数でも別q/Rは非同一。selectのschedule_menu.parameterizationとgroupは明示layout/共通κを分離する。旧API名inner_M_kappa_minimaの明示layout groupも**入力menu内の比較用区分**に限り、κ最適化の証拠ではない。

Dのprepare_full_h metadata validatorをBから再利用してone-body/全H λ/source/scalar/lawを照合する。入力descriptorにある`preparation_hash`や既存sourceを保全し、新科学preparationは実行しない。B3は受け入れない。layoutとcompiler/control/軸/full wrapper、`priced_equivalence_claim=false`を保全。既存外部対応表を利用し、新native physical hookはない。

`validate_requests`は**Dが生成した値未取得の外部request対応表**を検査する関数。科学N/C結果の資格判定は既存B.assess/E.planを使用する。physical ID単独・旧packetの付け替えは禁止。結果側は同一canonical identity/law/SOURCE/profileを満たし、full-wrapper単価・十分N・global契約を明示する必要がある。

## 検証と残作業

[TEST_RESULTS](TEST_RESULTS.json)は93件PASS。新結合テストは`python -B tests/check_abc_d_v2_integration.py`、D取込みテストは`python -B tests/check_full_baseline_v2_requests.py`。その他はTEST_RESULTS.modulesの各suite。evidence suiteは`test_explicit_menu_sources_cohorts_and_empty_inputs`だけを除外する（8次検証を含むため）。旧S0 producerを呼ぶsuiteは今回対象外。新CLI/adapterはmetadata専用。

今回Full PF/Full-H RTEの実比較menuは決めていない。Aの十分N/全RMSE、Cの適格平均価格/全law/長q、不確かさ・cross候補共分散、DのSOURCE付き物理layoutが不足。ユーザーが科学的条件と有限実行範囲を別途許可するまで新科学計算や実Gへ進まない。公開照合後**mandatory STOP**。
