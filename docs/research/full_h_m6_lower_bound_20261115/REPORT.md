# Full-H κ2 m6：保存証拠からの片側下界

**m6だけでp7 S2との順位は決まらなかった。正当化・数値化できた下界は `L6_lower=0 RZ`。基準593,555,352 RZまでの不足額は593,555,352 RZである。** この0下界は、固定ideal finite-K2 Full-H lawの全support、X/Y、測定付きfull controlled wrapper、指定compilerの出力に成立する非負性によるもの。PCG64の頻度やIID仮定を必要としない。非自明な下界が存在しないと証明したわけではない。

次に最も必要なのは、**合法な周辺event・境界がどう続いても、特定のRTE構造に由来するRZが固定compilerの最適化後に残るという不等式**である。現在の保存receiptは1 pathの正しさと価格を示すが、その不等式を示さない。全supportを結び付ける入力として、D7 m6の保存preparation hashに一致する完全descriptorが必要。新規compileが直ちに必要とは判断しない。

## 対象・正本

Linear H4 1.00 Å、STO-3G、保存tol-only DF10・H/state/snapshot。Full-H pure RTE（one-bodyを含む全非scalar H、決定論headなし）、κ2、finite K2。m6、T=6.4、q grouping=64、R=566、event位置数36,224、physical ID `FULL_H_4ec8d73571760a6830b8`。Bの固定1.5倍配分のX/Y各57 shotを保持する。状態準備P0。

scalar −0.33147781341681615をtotal Tで一回、new `CONTROLLED_ZZ_CX_CRZ_CX_V1`、symmetric-directional、full Hadamard X/Y測定wrapper。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、coupling mapなし。compiled RZは全回路のpost-transpile `rz`命令数。

| 内容 | 固定SOURCE |
|---|---|
| m6候補・law・N・元receipt索引 | `61f249e6e18d5afab9ba8797bef5e8f9f14e29de:docs/research/full_h_d7_holdout_admission_plan_20261115/ADMISSION.json` |
| D1最終holdout | `06956c64380107392f1b24fe54b9273f0f5ae1c0:docs/research/d_full_h_d7_holdout_20261115/` |
| 元compile実行 | `9cbabc617e95a81810924413c1b2b06ced329514:scripts/run_full_h_d7_holdout.py` |
| 元m6価格・implementation・compile record/raw manifest | `b5fac0ee4eac83f027f883e5ea5630623a98bf29:docs/research/d_full_h_d7_holdout_20261115/run01/M6_ROUND/` |
| Bの固定配分/Q3 | `6c5569c3479013b080b2ba53842585f1d670d3c9` |
| 比較基準 | `1002f1dfea34de439ca1b704a0192bca473c5387:docs/research/pf6_allround_cost_20261115/RESULT.json#rows[0]` |
| 凍結モデル・F1参照 | `b5165f11bb63f9eb2a6bfe385ba75d8003b7e724`、`a1aa38df46c92e2489095891f6082a5be1a277f5`。再fit/再予測なし |

[DECISION.json](DECISION.json) は下界・比較・候補/axisへの索引、[SOURCE_INDEX.json](SOURCE_INDEX.json) は実際に読んだSOURCE。旧F1のm6モデル部分額633,699,606.0515… RZと約4,014万RZの差は参照のみ。検証済み誤差予算にも下界にも使用しない。

元RTE71ba49・Bのlaw参照af284・D1 provider c31efの `_paired_order_weight`、`finite_rte_distribution`、`_make_event` はASTで一致した。これは定義関数のSOURCE対応であり、完全component descriptorの一致、I/Q分布一致、PCG64独立性の証明ではない。

## §10.5：再利用と今回の差分

関連C/D1公開branch、Git履歴、旧Full-H・価格検証・baseline設計・PF compiler監査を照合した。同等のm6母期待下界の公開結果/進行中jobは確認範囲で見つからなかった（自分のユーザーの関連Python job名に限定し、未知archive/他ホストの不在は主張しない）。他担当のworktree/private QPY/processを変更・探索していない。

`331a6ffebe551577eb9574aca354e76175697743:docs/research/rpe_baseline_optimization_design_20261113/BASELINE_OPTIMIZATION_PLAN.md` §4には、Full-Hの確認済み普遍的compiled RZ下界0と、event数×単価/構造proxyを下界へ昇格しない規則がある。今回これを再利用し、後発D7の同条件m6証拠で強化できるかを調べた。D7 fit・holdout比較・F1予測を繰り返していない。

本問は**固定shot配分**の費用である。57を与件として使えるので、量子shot数の必要下界や十分Nの新認定を追加条件にはしない。比較相手も元の採択済みcompiler閉包・既評価配分による条件付き基準を保持する。

## 実行経路と、各候補下界の扱い

| 経路/証拠 | 確認した事実 | 下界への不足 |
|---|---|---|
| Full-H preparation | `9b83…:scripts/h46_full_h_pure_rte.py:prepare_full_h` はone-body Zをtailへ追加し、deterministic_blocksを空にする | 決定論PFのone-body方向制御反復障壁がない。Full/p7のQPY閉包を移送できない |
| K2 law | `71ba49…:src/trotterlib/rte.py:1460-1491,1594-1659`。order0/2、component正規化、product→rotationの演算順・phase | 確率式だけではcompiled価格の事象上の最小値が得られない。rare order2を除外しない |
| providerのbasis再利用 | `c31ef…:src/trotterlib/df_rte_qiskit.py:278-350`。同basis連続ならbasis変換対を省略。phaseはrequest末尾へ集約 | naive basis変換数は既に計数前から相殺される。1 event当たり一定価格の普遍的下界ではない |
| new ZZ・中心操作 | `01e068…:efficient_controlled_zz.py:make_builder_class`、元provider`:472-502`。controlled ZZはCX/CRZ/CX、productはCZ | CRZやbasis primitiveのcompile前個数は、post-RZの保証された生存個数と違う。全外部continuationを含む生存証明なし |
| outer/scalar/wrapper | `af284…:scripts/h46_long_time_native.py:38-67`、wrapper `ax2b_native_df_v5.py:259-283`。tailを合成、scalar最後に一回、H/U/(Sdg)/H/measure | wrapperだけを個別compileした固定費やscalar由来RZを、全体最適化後の必残寄与と認定しない |
| compiler | `compile_cached:161-183` → `rte_compiled_cost.py:843-894`。full wrapper全体をtranspileして命令数を測る | RZ数の単調保存規則はない。opt1は1q再合成・identity除去と逆元相殺を反復する |
| D7 m6 receipt | 1 fresh pathのX/Y、元compile・fingerprint・scalar/event/XY gateをC受入済み。index0で元7数値gate | 1 pathの価格が高くても、別pathでの価格や母平均の片側誤差は保証されない |
| D7モデル/小holdout誤差 | 原モデルはCALIBRATED_UNVALIDATED、許容差null。m6は記述的比較 | 最大過大予測誤差・片側confidence・全support残差上界なし |

installed Qiskit1.3.0の `builtin_plugins.py:565-587`、`optimize_1q_decomposition.py:60-70`、InverseCancellation SOURCEを読取った。passや回路は実行していない。installed Python SOURCEのpath/hashはSOURCE_INDEXへ記録した。Rustの既存PF監査は参照tagとの対応に仮定を残し、そもそもFull-H全pathを監査していない。現在の0下界はこの対応仮定にも依存しない。

「最適化で全RZが消える合法m6 pathがある」とは主張しない。必要な非自明な生存不等式を今回確認できなかったというUNKNOWNであり、実際の最小価格が0だという結論ではない。保存QPY一対を追加で読んでも、別pathの全continuationへの保証にはならない。

## 旧Full-H確率情報を流用できない具体的理由

旧9b83の公開 `input_and_validation_summary.json#H4` はH/state/snapshot・scalarが一致する一方、preparation hashが `0e3989e739d0e17a5d7af0fe809bd7c10ee765f09c934fe6ebb1f0e8b5c1046d`。
D7 m6は `92325333cfe8c5e6c5f9bf69cdfc0ea6538f8eaa66327a3820d390dc2e8aa100`。

旧summaryの368 componentは旧hashのmetadataであり、今回の完全support表として受け入れない。差の原因は未同定。旧価格はK4/q16/R16/T0.8で、`population_tail_cost_bound=null`、PRNG独立性未証明だった。新ZZ・m6・K2の片側評価へ移植しない。この保留は旧K4資産の移送に関するもので、D7自身のhashへ固定されたreceipt受入を取り消すものではない。

PR-DFの324 component・F0距離や構造E[S]も別domain。Full-Hのcompiled価格に対する全pathの `C−S` 片側残差上界は確認できない。したがって構造期待値や局所connected-cluster補正から、今回のcompiled価格下界を作らない。

## 確率が既知の事象を使う道と現状

事象A上で全path共通に `C_6X+C_6Y≥h_A` が証明されれば、非負性から `L6≥57 h_A P_I(A)`。複数事象を足すなら、互いに排反な分割又は二重計上しないpointwise報酬不等式が必要である。

現在不足するのは `h_A` のpost-compiler保証と、その同一preparation・lawへのbinding。order0だけ、basis switchだけ、保存pathだけの価格を `h_A` に代入できない。ソース上の確率式を再評価するだけではこの不足は埋まらない。

## 導出と比較

compilerの出力に対し `C_a(ω)=#{instruction.name==rz}` は非負整数。よって全合法ωでC_X+C_Y≥0、任意の確率法則Iに対し

`L6=57 E_I[C_X+C_Y] ≥ 0 = L6_lower`。

| 固定対象 | 値RZ | 判定 |
|---|---:|---|
| 正当化できたL6_lower | 0 | 全supportの非負性下界 |
| p7 S2固定全額 | 593,555,352 | 元の採択条件を保持 |
| 基準−下界 | 593,555,352 | 正の不足額。順位未判定 |

上回るために必要な軸和の母下界は `197851784/19 RZ` より大きいこと（F1保存閾値を再利用）。現在の0下界が小さいことから、Full-Hの実費用が安いとは推論しない。

## 次の有限案・終了

[NEXT_PLAN.md](NEXT_PLAN.md) / [NEXT_PLAN.json](NEXT_PLAN.json)：先にD7 hashに一致する保存完全support descriptorの回収・bindingと局所生存不等式を検討する（新path/compile0）。新しい価格取得が必要なら、事前固定した価格閾値の成功事象に対する片側検定案を提示した。提案パラメータでは最大61 fresh path・122 wrapperだが、I/Q事象確率差と独立性条件・price confidenceの採択が先であり、現在は実行不可。これらの条件を解決せず追加compileを先行させない。

今回新規sampling/build/compile/fit/Q3、予測/holdout再計算は0。実装の新設なし。既存readerとSOURCEを再利用し、保存結果のmetadata・非負性算術・未採択案のexactサイズ条件だけを確認（[CHECKS.json](CHECKS.json)）。新CIや母価格値は作成していない。

ロードマップ差分：m6単独のp7比較は未判定。全supportで正当化できた下界0、母期待部分和のpost-compiler証拠が不足。旧Full-H preparation差を明示し、次の入力をD7 hash一致descriptorに限定。F1全額null、D7旧資格・原結果・STOPは不変。ロードマップ本体は統合担当へ。公開照合後mandatory STOP。
