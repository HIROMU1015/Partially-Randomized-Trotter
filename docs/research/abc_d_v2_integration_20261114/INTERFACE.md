# B → D：標準4次PF・明示Full-H layoutの共有仕様

2026-11-14。本commitは**interface先行公開・実装/検証は後続**。科学的baseline・RMSE保証・κ等価性の採択ではない。B基点030f3f548、D原版eecaa8030280f8957ad85555bcc89e2daa1a29faを継続利用する。

## 標準Yoshida4

- `spec.method='PF'`、`formula='4th'`。Full PF/TRUNCATED_Hの区分と元H/保持Hを従来どおり保全する。PRの標準4次は今回の拡張対象にしない。
- `pf_pr_rpe_pipeline.structure`は既存`h46_m5_plan.weights('4th')`のbinary64係数・順序・負時間を使う。
- formula_idは`PF_STANDARD_YOSHIDA4_BINARY64_V1`。`PF_M5_BEST_20D64C2_BINARY64_V1`とは別。PFはκ/Kなし。
- 既存formula-menu指定と比較条件を維持。8次の接続・検証は追加しない。

## κなし明示Full-H RTE

- Bの定数`EXPLICIT_LAYOUT='FULL_H_EXPLICIT_LAYOUT_NO_KAPPA_V1'`を追加する。
- このkindは`method='FULL_H_RTE'`、`kappa=None`専用。入力K・M・q/R・H/state/preparation/law/profileを変更しない。既存κ由来/κ付きSOURCE mappingは従来のkind・κ定義を維持する。
- Dの`full_h_rte_spec(base, M=..., finite_K=..., kappa=None, preparation=...)`は上記kindを使う。κを供給した旧呼出しは従来kindのまま。
- Dは`generate(spec, layout_by_round={m:{q, R_outer}})`から既存layout_mapperを渡す。Bは物理lawやq/Rを生成しない。
- round mappingはkind/source/`kappa=None`、`R_outer=None`、`event_count=q*R`と、Dの`physical_layout`全体を保全する。layoutのq/R、event_positions、allocation、K、scalar一回、deterministic_blocks=0、`priced_equivalence_claim=false`を照合する。
- 全H preparationはDの既存validatorでone-bodyを含む全非scalar項・全H λ/SOURCE・scalar/lawを検査する。Bはそのsource-bound descriptorとsplitの整合を保全する。B3の転用は不可。
- layout未指定roundは未評価のまま、`UNRESOLVED_PHYSICAL_LAYOUT`。Bは架空のκやlayoutを補完しない。
- 同event数でもq/R groupingが異なれば別configuration/候補ID。明示layoutと論文κの等価性・最適性・RMSE達成は主張しない。

## 物理候補の接続

D既存`requests(config)`の外部対応表を使用する。native controllerへの新physical hookは追加しない。

各A/C requestで`configuration_id/v2_candidate_id/identity/law_binding/physical_candidate_id/physical_candidate/physical_schedule_mapping`を一致させる。元physical ID単独は再利用keyではなく、全H/state/SOURCE/profileと全schedule contextを一緒に保持する。旧recordへ新IDを付けない。

新`D.validate_requests(config, packet)`はこの外部対応と全identityを検査するmetadata helperとする。Dが生成した対応表はN/C取得や資格の証拠ではない。

## 固定S0の取込み

B公開030f3f548の保存V2 configurationをそのまま読み、A75b5c89の公開packetを既存`validate_A_v2_packet`で受入検査する。C d458b5eの保存PRICE_REQUESTSを既存`verify_B_publication`と全26B requestへ照合する。A.connect/C.prepare/H4 S0.prepareを呼ばず、S0を再生成・再解析しない。nominal64/38、他11round未評価、十分N/global bound/C/Gのnull、C1 falseと有限cohortを保全する。

実装・人工/関連回帰の後に最終REPORT/HANDOFFを公開する。新科学実行・compile/fit/実Gは別の明示scopeが必要で、今回の最終公開後mandatory STOP。
