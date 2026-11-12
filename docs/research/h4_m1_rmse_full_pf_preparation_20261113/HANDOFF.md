# A → B/D：m1保存診断の準備とFull PF条件

2026-11-13 JST。実科学計算・保存値数値再解析0。m1は準備のみ。

## Bへのm1接続

- 固定B030f3f548a35d98a88e62e92a35f132e0243a293の正本と、A75b5c89b…V2 configurationは完全一致。IDは`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。
- B正式`validate_A_v2_packet`で既存S0 packetを受入確認。m0/m6はnominal64/38、他11は未評価、全十分N/global bound=null。元packetは変更なし。
- 対象m1はT=.2/qPF2/R4/physical `c00e00dc1e9c414fb7d9`。mean/pairは公開89c91c692…`run01/result.json:records[5]`、元科学SOURCE90dcc3ed2…。
- 新`h4_pr_m1_rmse`は固定S0 scalar gate・既存R-aware allowance部分・checked_reference・geometryを再利用。旧analyze/Hoeffding solverを呼ばない。geometry出典は実H4 batchを使用し、P1 carrierへ誤帰属しない。
- [PREFLIGHT](PREFLIGHT.json)はfield/metadataだけで、m1新nominal・診断は未評価。default CLIに科学callなし。

将来のm1 packetは、canonical全13round identity/lawを保ち、m1の新診断SOURCEとm0/m6の元SOURCEを各rowの`diagnostic_source`で分離する。`preserved_S0_packet`で元packetを参照し、`diagnostic_scope.newly_reanalyzed_rounds=[1]`を付ける。十分N/bound/global資格は未取得のまま。

**現BのS0-only validatorはm1値を拒否する仕様**。m1をS0許可/packetに偽装しない。Aの準備済みhandoffは既存B `n_from_a`を再利用するが、新m1公表後の専用受入scope更新はB担当。B共通planner/identity/C record APIをAでは変更していない。

## 承認が必要な次scope

**H4 PR m1一候補の保存scalar再解析のみ**：保存pair/gate・raw/corrected/exact-tail/norm/closure/bias・R-aware allowance・局所phase/radius・経験的nominal X/Yを評価し、未資格V2 packetにする。新mean/対照/reference action/normalizer/array load/decoder simulation/価格/Gは0。m0/m6はcopyだけ。

実行には別途ユーザーの明示承認と、今回公開したmodule/runnerの固定科学SOURCEが必要。CLIは`--execute --source <固定commit SHA> --out <新規directory>`。新grantや承認証明書を要求しない。不足/不一致/FAIL/非有限値/丸め・gate未解決はSTOP、fallback/retry/候補変更なし。

## D/BのFull PFとの接続条件

[Full PF SOURCE catalog](FULL_PF_SOURCES.json)／[詳細表](REPORT.md)。D固定eecaa803…のspec builder/catalogを再利用。実Full PF configurationやNは採択・生成していない。

- Fullは全one-body＋全DF fragment、H4 rank10/H6 rank19。B0/prefix7と区別。R_outer/κ/K/λ_R不要、deterministic、event mappingなし、scalarはham.constantをtotal Tで一回。設計上B=1だが、stateと有効PF固有stateの不一致による信号振幅低下を無視しない。
- S2とm5_bestはB V2生成/A物理action経路がある。標準Yoshida4は元3 signed pieceのsignal SOURCEがあるがB V2/A common.structureの接続待ち。Aではm5へaliasせず、物理展開を再実装しない。8次対象外。
- D requestのphysical candidate IDとV2 ID・actual幅を正本とする。A旧scheduleはPR＋旧B0対・κ/tail必須なので、Full PFへそのまま使わずPF-only接続を整える必要がある。偽κ/Kを入れない。
- 136保存carrierをfieldで索引化したが、元norm観測scalarやallowance/pairの欠測・SOURCE/実幅不一致は未解決のまま。全てを評価可能/合格とはしない。新数値診断0。
- PR policyのcopyやB=1の形式的shot則から、Full PF用の十分Nを認定しない。元DF-H groundと有効PF energy/state、TrotterとQPE誤差、prior/alias/全decoder統計の契約が必要。

**次の担当**：ユーザーはm1の上記scopeを判断。Aは承認後だけ実行。Bは将来m1診断packetの新scope受入、B/DはFull PF/標準Yoshida4の共通接続を担当する。Full PFの公式/幅/M/policy・科学実行は別途ユーザーが決める。公開照合後mandatory STOP。
