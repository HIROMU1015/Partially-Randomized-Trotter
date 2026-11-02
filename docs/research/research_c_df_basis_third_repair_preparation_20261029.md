# 第三γ0 DF：components接続修正とfresh run02準備

2026-10-29 JST。起点`7d5536f469800f92b248242af9288728d2ed8b1c`。ユーザーは修正と再実行を明示許可した。旧[141-call STOP](research_c_df_basis_third_execution_20261029.md)のSOURCE・raw・契約・STOPは保全し、残枠/resumeとして扱わない。

## 修正範囲と再利用

新`c_df_basis_third_v2.py`は、既存B1/P4のcomponent-recordと同じcomponent_id/probability/coefficient_abs/coefficient_sign/basis_id/df_fragment_idを、既存`tail_gate`の戻り値へ補完する。値は既存`tail.components`の写しで、確率の再正規化・sampling・符号・lambda・phase・gate条件は変更しない。矛盾する既存metadataはFAILとし、fallbackしない。

旧SOURCE112件はbyte単位で不変。追加はmetadata adapter／v2 CLI／人工testの3ファイルだけである。旧一体engineのstage_referenceを専用process内で包み、law_prepareから既存tailを受け取り、同じtail_gate logical unit内でrecordを補完する。Scope/version/fresh pathだけをv2へ結合し、終了時はfinallyで旧bindingへ戻す。数値engine・CAR・PF/RTE・RPE・native・P0・モデル訓練/判定は旧関数をそのまま使う。

## 人工・回帰

**新4＋既存42＝46 PASS（11.42秒）**。[ログ・検証](../../artifacts/research_c_df_basis_third/2026-10-29/technical_repair_v2/)。

- 新第三DFの人工reference出力を、mockしない実B2 reference_context/law_gateへ接続してPASS。
- componentsを除くと旧KeyErrorを再現。確率列を変えると既存gateが拒否し、矛盾metadataも拒否。
- Reference/native32/holdoutの493-call人工dispatchでも、独立full CARのlaw gateを実際に通過。他の高価な回路価格/operator部分はmock。
- Bindingの例外時復元、科学条件・旧入力の無断変更拒否、旧外部検証/一般K/run/key回帰を確認。

人工テストの不正fixtureを同じexclusive-createファイルへ二度保存する不備を修正した。科学許容差は変更していない。旧ffsim/codec/AD部分は修正しておらず、準備時の131 tests証拠を保全した。今回の46 PASSは実H4のgate PASSではない。

## 新規固定と実行範囲

[修正契約](research_c_df_basis_third_contract_repair_v2_20261029.json)は旧契約SHA・科学条件・モデル・scenarios・compiler・sampling・call上限・原科学input93件を照合する。新SOURCE115件、保全確認用を含むinput142件、旧保護1725件を固定した。

- 契約SHA256：`878f87d8769621a491bb99dc5f8880a9f93dcb7f58d4749f7652c6bda407846b`。
- SOURCE manifest SHA256：`9790f3f1342dd9fa993b08e0985ebadf9c4103c914a1d35451b4c17a658bd006`。

保存直鎖H4・1Å/STO-3G、γ0 number CDF8/元順LD2、Nα=Nβ2/sector36、δ=h=.1/s1/r128/K0/m0..12、CA=.00159 Ha/DF reserve CA/8、元g補正κ/scalar一回、own-DF oracle/状態準備除外は不変。共通interceptを第三DF価格より前に固定し、旧Cを復活させない。

新fresh outputは`artifacts/research_c_df_basis_third/2026-10-29/run02`。上限493 unitsを一度だけ新しく実行する。既存一体runnerと登録順序を保つため、T/A/RPEからfreshで実行し、旧141-callの部分resumeや原本修復はしない。Technical relaunch=1、run内retry/resume=0、新fit/optimizer/rank探索/DF誤差再計算0。Call定義は[旧設計](research_c_df_basis_switch_optimization_design_20261029.md)の8/64/48/8/357/7/1を維持し、metadata補完は既存tail_gate内で行う。

Samplerのmaster/cohort/PCG64政策は同じで、元の規則どおり新SOURCE digestがseedへ入る。旧run01はsampling前にSTOPしており、観測価格に基づくcohortの選別はない。新実行でもcorrectness FAILは後続を止め、poor transfer/未資格を条件変更で救済しない。全law synthesis/law差/cost support/actual CA/母平均・因果効果/η採択は未資格のまま。

修正SOURCEをcommit/pushしてremote照合後、チャット許可に基づき実行する。`unapproved`等の歴史的metadataを承認証明へ変更せず、証明書・新grantは作らない。旧同規模の実測は約920〜1038秒で、支配処理はnative32 full512/compile/adjoint検証。既存の資源に応じた13/32並列・Qiskit排他を再利用し、人工的なCPU/RAM/worker/wall capは追加しない。共有依存read-only、cache/task出力だけを変更する。

科学結果と旧新原本を公開照合後mandatory STOP。新DF目的・η・追加fit・別分子・詳細GPT科学レビューへ自動進行しない。
