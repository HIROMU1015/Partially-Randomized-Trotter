# H4 DF10 PR m1：保存RMSE局所診断・nominal N結果

2026-11-13 JST。**承認されたm1一候補を固定科学SOURCE ac873e7fcc42e937084fa153476cbd87565702b4のrunnerで一回実行し、保存数値gateを通過した。経験的nominal N_X=N_Y=60。** modelは`UNQUALIFIED_MODEL_ASSUMPTIONS`、十分N・全RPE RMSE boundはnull。新mean/reference/normalizer/H-state array load、sampling/decoder simulation、Full PF、C/Gはすべて0。

[科学結果](run01/result.json)／[V2診断packet](run01/N_HANDOFF.json)／[SOURCE索引](SOURCE_INDEX.json)／[実行出力](EXECUTION.txt)／[事前metadata検査](PREFLIGHT.json)／[最新B互換性](B_V2_COMPATIBILITY.json)／[packet検査](PACKET_CHECKS.json)／[技術テスト](TESTS.txt)／[B向けHANDOFF](HANDOFF.md)。

## 固定条件・SOURCE照合

Linear H4・隣接1.00Å・STO-3G、保存tol-only1e−8 DF10/completion/元fragment順・scalar、保存DF-Hのsector(2,2)/dimension36 ground energyを対象とする保存数値state条件付き評価。S2/L_D1/ΔPF=.1/K2/h=.1/M12/P=0、無energy shift。εE=.00159360010199040 Ha、εqpe=εtrot=.00079680005099520 Ha、minkowski/finite_B_diagnostic。共通κはS0 CONFIGURATIONのexact有理数を保持し、表示小数から再生成していない。prior・decoder・law・compiler/control/full wrapper/profileを変更していない。

| 項目 | 今回の固定値・出典 |
|---|---|
| 科学SOURCE | **ac873e7fcc42e937084fa153476cbd87565702b4**、既存`h4_pr_m1_rmse.py` / `run_h4_pr_m1_rmse.py`をbyte無変更で使用 |
| 元保存carrier | 89c91c692e1d35de94d2e7afbe890aabe6d80b7e、`docs/research/h4_df10_rpe_batch_results_20261110/run01/result.json:records[5]` |
| 元m1数値SOURCE | 90dcc3ed20f30dfab6b22a68cc4adb9461485a98、H4 batchの保存finite mean/reference |
| scalar gate / R-aware式 | 7c3e1e5db0c55dd8249e9030a4c7e9954496fa7eの固定S0 gate・allowance prefixのみ。旧`analyze`全体/Hoeffding solverは呼ばない |
| reference gate | 71ba49bb28b295aa28e51101bc3b1bf07d500767の既存`checked_reference`。carrier出典はm1のまま |
| 元S0 | 公開7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75、m0/m6 model recordはcopyのみ |
| 最新B | **f0c4bb559d2a64dabe88e2eba31ff1dadad6fca8**。正本の起点030f3f548a35d98a88e62e92a35f132e0243a293と同canonical PR configuration |
| V2 ID | `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911` |
| 候補 | **m1/T=.2/qRPE2/qPF2/R_outer4/8 event位置**、physical `c00e00dc1e9c414fb7d9` |

H/state/snapshotの完全hash・profile・消費した保存JSONのhashをSOURCE_INDEXへ記録した。binding内のT=.8は元reference由来の歴史的metadataで、今回の発展時間はcandidateのT=.2。bindingを書き換えて一致を作っていない。

B最新の更新は標準Yoshida4/Full-H layout等を含むが、今回のPR正本・A integration・S0 intakeとcanonical assembly/validationは固定版と同一。Aの科学SOURCEをB最新へ付け替えず、最新B正本との全13round identity/law対応を別途metadataで照合した。Full PFの新機能は今回使用しない。

## 一回実行と数値gate

```bash
python3 -B scripts/run_h4_pr_m1_rmse.py --execute \
  --source ac873e7fcc42e937084fa153476cbd87565702b4 \
  --out docs/research/h4_pr_m1_rmse_results_20261113/run01
```

これは完了した実行の再現情報で、同じ出力先への再実行や新科学callの提案ではない。入力・結果の不一致やFAILは発生せず、科学retry・候補/閾値/κ/M変更は0。ユーザーのチャット承認をそのままscopeの許可として扱い、承認証明書・新grantは要求していない。

| 保存数値gate | 保存/今回確認値 | 判定 |
|---|---:|---|
| 二経路reference差 | 5.551115123125783e−17 | ≤1e−10 |
| exact-tail norm誤差 | 0.0（元保存の観測値） | ≤1e−9 |
| 独立raw/corrected vector closure | 5.962499244401586e−16 | ≤1e−9 |
| scalar raw/corrected closure | 5.578801654593729e−16 | ≤1e−9 |
| 最大中間norm | 1.0000000000000002 | 有限・元gate成立 |
| bias分解・allowance・丸め | 元gate成立、ceil精度安定 | 数値診断通過 |

reference内の`reference_state_actions=2`、`kind=NEW_D2_TWO_EXPONENTIAL_PATHS`は**元H4 batchに保存された過去の取得情報**である。今回新reference actionは0。未知のactual law/ground/実装誤差は観測norm0と別で、nullのまま保持した。raw signalをcorrected/Bから事後生成していない。

## signal・normalization・allowance

以下は今回resultに保存した値の表示丸め。各複素値の完全精度はresult参照。

| signal | Re | Im |
|---|---:|---:|
| full-H reference（保存path1） | .9075945803256485 | .41984768400398476 |
| independent reference（保存path2） | .9075945803256485 | .4198476840039848 |
| corrected | .907605831516938 | .41982334932824544 |
| raw | .8712704374469875 | .4030160015701193 |
| exact-tail内部対照 | .9076058334371134 | .41982335023166323 |

| 指標 | 値 |
|---|---:|
| 保存log B | .040857758676500454 |
| B（保存logからのexp表示） | 1.0417039216622817 |
| attenuation | .9599656670239531 |
| 絶対corrected bias | 2.6809806951348577e−5 |
| 絶対finite-K RTE bias | 2.122083293205865e−9 |
| correctedのreferenceに対する位相 [rad] | −2.680980655681611e−5 |
| rawのreferenceに対する位相 [rad] | −2.6809806556937006e−5 |

finite normalizerの分布再計数はない。元SOURCEの式でR-aware allowanceを適用し、旧allowanceを保全して`max(previous,R-aware)`とした。適用後u=**1.0000000555111513e−9**、logB margin=**1e−12**、stages=26、roundoff項=7.673861546209082e−13。保存previous u/marginも同じ値であり、元carrierと適用式のSOURCEを分離して保存した。数値allowanceはempirical/certified=false、rigorous boundではない。

## 位相・半径・経験的N

| 局所指標 | 値 |
|---|---:|
| corrected信号半径 | .9999999950223691 |
| raw信号半径 | .959965662245599 |
| 条件付きcorrected disk | 2.681113430169533e−5 |
| 条件付きraw error disk | 2.5737769384145032e−5 |
| 条件付きraw半径下界 | .959939929254569 |
| βdet [rad] | 2.6811135305515427e−5 |
| **経験的nominal N_X / N_Y** | **60 / 60** |
| model status | UNQUALIFIED_MODEL_ASSUMPTIONS |

Nは固定M12と保存finite logBによる経験的`ceil(exp(2 logB)[11+4(M−m)])`。既存Decimal複数精度のceil安定性が成立した。旧高確率型Nの流用ではなく、十分shot数やactual minimum Nではない。

条件付きdiskは元S0と同じ保存数値state/residual・empirical u/marginのrecipe。E_stateはreference診断だけに使い、prior/decoder branch選択には使っていない。上表のβdet/半径や局所biasをenergy RMSE/εtrot上界へ換算していない。actual implementation/law/厳密ground同定の誤差・actual radius/β・RMSE boundはすべてnull。

## V2 packetと残る証拠

[N_HANDOFF](run01/N_HANDOFF.json)は新scopeを`newly_reanalyzed_rounds=[1]`、`preserved_S0_rounds=[0,6]`とする。各rowで元physical ID・canonical V2 identity/law・diagnostic_SOURCEを保全した。

| round | nominal X/Y | 出典 | 状態 |
|---|---|---|---|
| m0 | 64/64 | 元S0をcopy | UNQUALIFIED_MODEL_ASSUMPTIONS |
| **m1** | **60/60** | **今回SOURCE ac873e7fc…** | UNQUALIFIED_MODEL_ASSUMPTIONS |
| m6 | 38/38 | 元S0をcopy | UNQUALIFIED_MODEL_ASSUMPTIONS |
| m2/3/4/5/7/8/9/10/11/12 | null/null | 新評価なし | NOT_EVALUATED |

全13roundの十分N_X/Yとglobal RMSE boundはnull、全体/finite-K/signed mappingはUNQUALIFIED。BのS0-only validatorへm1を無理に通さず、元S0 packetは変更しない。新scopeの受入はBへHANDOFFする。共通C API・価格・全Gは変更/計算していない。

残る局所診断は10round。m3/m4には一致保存meanがあるが観測norm scalar/legacy backend allowance bridgeと独立reference pairの課題が残り、m2/m5/m7–12の8候補は一致mean欠測。今回これらを新たに数値診断していない。

全RPE RMSEには、finite-K meanと原paper統計モデル、保存stateと有効発展/ground、Minkowski成分誤差、prior/alias/branch、全13round decoder統計、未解決試行loss、sampling law/実装誤差の証拠が必要。二点S0にm1を加えてもこれらの資格は成立しない。

## 検証・保全・STOP

既存m1人工/metadata11＋元S0人工7＋RMSE policy人工17＝**35テストPASS**。新科学用テストframeworkや旧N solverは追加しない。最新Bとの全13round metadata・S0 nominal/source保持・残10null・十分N/global nullを機械照合した。元科学code、S0/V2/raw/入力/過去STOPの不変も確認した。作者自身の技術/数値確認で、独立科学レビューではない。

今回のscopeは保存候補1・局所診断1・nominal X/Y一組。新mean/対照/reference action/normalizer/array load/trajectory/decoder simulation/compile/C/Gは0。Full PF・他round・κ/M最適化・詳細科学レビューへ進まない。

**次の担当**：**Codex B**はこのm1追加scopeの未資格packetを同canonical IDで受け入れる条件を整える。**ユーザー**は残10roundの局所証拠補完・全RMSE検証の次の有限scopeを判断・承認する。**Codex A**は別途承認後にだけ実行する。今回はcommit/push・remote SHA/主要成果物照合後**mandatory STOP**。
