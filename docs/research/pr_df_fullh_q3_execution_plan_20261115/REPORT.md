# PR-DF LD1/LD2・Full-H κ2：保存13roundのQ3実行計画

2026-11-15 JST。既存B branch `codex/rpe-total-dp-acceleration-20261114`、基盤 **8dba096dc39788b8948af9eafe394f03d386c520** を継続使用。今回の作業は固定資料の読取・重複照合・計画公開だけ。**新Q3/DP/二項確率/MC/shot探索/価格/Gは0件**。D1の価格較正はこの精度側計画の依存条件ではない。

## 3候補の結論

共通decoder D＝`consecutive_dyadic_nearest_branch_principal_energy_total_v1`（Z1＋T1）。P1/P2/Hは下記固定packet。IDは元configuration→total configurationの先頭を示し、完全なID・binding ID・全39roundのphysical ID・law/SOURCE参照は[CANDIDATE_INPUT_INDEX.json](CANDIDATE_INPUT_INDEX.json)が正本。

| 候補 | 固定packet・decoder/ID | 推奨shot vector（m0〜m12、X/Y別） | 同等Q3結果の有無 | 再利用／新規science call案 | 判定できること／実行前の不足 |
|---|---|---|---|---|---|
| PR-DF LD1 | P1、D、a4dda8f7→9c3d392e | X=`[64,60,56,52,47,43,38,34,30,25,21,17,12]`、Y=`[64,60,56,52,47,43,38,34,30,25,21,17,12]` | 確認した固定資料・関連履歴では未同定。旧partial DPは別推定量 | 13信号・SOURCE付きfixed-grid bridge・control配分を再利用。リスク不足分だけ最大1件 | この1vectorの直接モデルMSE対τ。閉包・total DP範囲の候補別確認、専用callerと完全return/資源保存の人工検証、科学実行許可が必要 |
| PR-DF LD2 | P2、D、1fa33556→e6fe7e68 | X=`[64,60,56,52,47,43,38,34,30,25,21,17,12]`、Y=`[64,60,56,52,47,43,38,34,30,25,21,17,12]` | 同上。LD1/Pauliの値は流用不可 | LD2固有13信号・bridgeを再利用。リスク不足分だけ最大1件 | 同上。元κ・R/allocation・normalizerはLD2固有のまま保持 |
| Full-H pure RTE κ2 | H、D、54ac3eb3→265b90dc | X=`[64,60,56,52,47,43,38,34,30,25,21,17,12]`、Y=`[64,60,56,52,47,43,38,34,30,25,21,17,12]` | 13信号とtotal metadata受入はあるが、同条件Q3は確認範囲で未同定 | κ2固有13信号・既存baseline bridgeを再利用。リスク不足分だけ最大1件 | 同vectorでのFull-Hモデル診断。閉包/範囲とSOURCE再照合、caller検証・科学許可が必要。R448は対象外 |

**将来の新規評価は最大3件であり、3件の実行をノルマにしない。** 実行直前の§10.5再照合で完全同等の結果を受理できればその候補を除外する。証拠不足なら該当候補を保留する。今回は実行を許可された計画ではない。

## 固定packetと物理条件

- **P1/P2**：A **714247619afb4714fd663eb15c0470ef32bdc7af**、`docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD1.json`／`B_MEANS_LD2.json`。元configuration **a9abf8de75bf3432fcd714e868e5c14c8884b8f8** の `docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json`／LD2。数値SOURCE **fefd548ecd56f60d72031b3d385f6f5e6ee95d06**、raw carrier **16210932e5dabe7d12d41f35fea0e0a0d7a9b271**、`run01/LD1/m<m>.json`／LD2。bundleは13複素meanを含む既存A形式で、C向けpacket viewは既存K受入器を使う。
- **H**：**ce6aacfaca8249c510baa2d958849cffc7649cdc**、`docs/research/d_full_h_kappa2_s1_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/PLAN.json:configuration` と `B_SIGNAL_INPUT.json`。原記録は同directoryの`round_<m>.json`。producer **5ccc1dfc341236d3fce2f745e7e9023447a027a7**、数値実装参照 **5cc7c07e112ce97205948465c94441cba0042416**。sampler準備SOURCE **9b83b96b83f62142c9d87858dc24a94db7ac5d61**。

共通：Linear H4/1.00 Å/STO-3G/tol-only DF10、sector(2,2)・36次元、保存full DF-H/state/snapshot、h=0.1/M12・全13round。H/state/snapshotの識別子と元prior/energy shiftは索引の固定configuration/targetに保持。

PR-DFはS2/要求幅0.1/K2、元のexact κと旧fixed-gridを維持する。9c0e12d87の版付き`A_FIXED_GRID_510FBACD_TOTAL_DECODER_VIEW_V1`を使い、元`A_FIXED_GRID_510FBACD_V1`のspecを保全。Pauli辞書・Pauli κ/R schedule・generic ceil-gridへ変換しない。元physical IDと新canonical candidate IDは別物。

Full-HはK2/κ2、one-bodyを含む全Hのpaired finite-Taylor law。canonicalのqPF/R_outerは非該当の表現を含む一方、物理requestのq/Rはgroupingとevent layoutを表す。元packetの`family=FULL_H_PURE_RTE`・one-body randomized・event数を保持し、物理builderに残る`method=PR/formula=2nd`という表示からPR-DFや決定論PFへ分類し直さない。R448・DF B3はこの候補ではない。

## normalization・scalar・target

3候補ともlawのmodel名は`ideal_finite_RTE_law`だが、辞書/準備・split・κ・event schedule・normalizerは候補別であり、同名lawだけで同一視しない。

PR-DFのscalarは保存Hamiltonian constantとprefix-tail scalarを総Tで一度適用する元producerの規則。tail scalarの保存値はLD1 `0.4177175120626939`、LD2 `0.2223759094107018`。出所は7a8bc7f1のSOURCE_INDEXが指す固定`H4_context.json:prefixes/1,2/scalar`で、既存hash一致を読取確認した。constantは同INDEXの保存snapshot array `constant`を出所とし、今回その和や位相は計算していない。fefd548eの`evaluate`と各元rawの`scalar_once=true`が適用証拠。Full-HはPLAN内の保存scalar `{constant:2.29310124732, extracted_identity:-2.6245790607368162, total:-0.33147781341681615, application:ONCE_AT_TOTAL_T}`をそのまま保持する。

normalizationは各meanの保存`log_B/B/attenuation`と数値allowanceを使う。例として保存m12のBはPR-DF LD1 `1.042001446266762`、LD2 `1.0419924691803404`、Full-H κ2 `1.6487196474611683`（転記のみ）。これを同じ減衰とみなさず、controlの共通化から必要Nの共通性や閾値適合を推論しない。rawはすでにこのモデルの減衰を含むので、correctedへ置換したりBをshot・loss・価格へ二重乗算しない。

評価targetは保存DF10 solver proxy **−2.1663874486347634 Ha**。正本はbba150834の`PHYSICAL_INPUTS.json:target`、truth proxyの元solver記録は89c91c692のbatch結果。元packetは`energy_ha`を持たず、ground認証false・proxy採用false・UNQUALIFIEDを保つ。将来のcallerでは採択済みproxyを**別の明示evaluation_target**として渡す。元packetを書き換えたり厳密groundへ認定したりしない。

## 配分推奨の根拠と限界

共通controlはe7ce23803の`h4_dp_risk_fixed_results_20261114/INPUTS.json:cases/*/allocation`に保存された実際のPR-DF旧partial評価用vectorをそのまま再利用する。Pauli元controlでも同形が使われたが、今回の出所はPR-DFの保存配分である。Pauli3/4や決定論PFのpaper-shaped vectorを流用した案ではない。

PR-DF LD1/LD2でphysical inputと配分を保持すれば、推定量変更の診断を配分変更と混同せずに始められる。Full-Hにも同じUSER_SUPPLIED配分を使えば、同じh/M/shot policy/decoder/targetでまず評価の可否を比較できる。Full-Hでは既存の**構成固有nominal/sufficient N**として保存された配分ではなく、別方式の既存control形状を新たな診断に用いる提案である。減衰差を補償できるという予測ではない。

同じshot vectorは公正比較の必要条件ではない。これは最初の1本を選ぶための診断上の利点で、後続の方式固有・cost-aware配分探索を排除しない。shot総数最小、有限menu内選択、全空間最適N、物理的十分Nは今回求めない。適格E[C]がなくてもこのモデル診断は計画できるが、G比較はできない。

## 重複照合の範囲と結果（§10.5）

固定C0b4b2d94 ALIGNMENT、B9c0のreadiness/PR-DF canonical mapping、旧DF DP e7ce2380、Pauli control caf1417と3/4評価50be68b8、PF6結果8dba096dを照合した。関連B/C/A/Full-H signal/decoder branchのlocal履歴と公開refも確認し、指定3候補の同一vector・decoder・target/law/lossに一致するQ3結果は確認範囲で未同定。これは全サーバー・全非公開記録の不存在証明ではない。

| 既存成果 | 今回の用途 |
|---|---|
| 39保存roundとその原record | SOURCE/physical ID/identity/law、raw/reference、normalizationの一致を読取確認。Aの信号取得から除外し、そのまま再利用 |
| PR-DF旧partial DP | 同じraw/旧controlの出所と旧decoder挙動の参考。totalのzero HOLD/tie argmin・全事象lossとは違うためMSE値・未解決lossをQ3へ転用しない |
| PR-DF fixed-grid/Full-H total metadata受入 | 既存ID・source-bound API接続を再利用。MSEを評価した証拠ではない |
| Pauli元control/3/4 Q3 | total kernel/backend・保存形式・有限scopeの技術的前例。辞書、physical input、κ/R/normalizer又はvectorが違い、3候補のMSEとしては不採用。再実行しない |
| Full PF/p7の6評価 | 別physical候補・別vector。モデルMSEは今回へ移送しない。監視caller修正と保存障害の教訓を再利用。CPU/peak RAM/API return欠測を埋めるretryはしない |
| Full-H R448 | κ2と別configuration/physical layout/raw/law/normalizer。対象外 |

自ユーザーの関連Python runnerをread-onlyで確認した範囲では対象のQ3 jobはなかった。`run_full_h_kappa2_p2.py`のprocessを観測したが、固定callerの説明で価格P2（sampler/trial/ZZ等）の処理と確認でき、Q3重複には数えなかった。D/D1のprocess・作業領域へ介入していない。

計画用索引は標準ライブラリの固定JSON読取・転記だけで作成した。39原record/78軸参照、3候補のlaw/split/H-state binding、元整数vector、ローカル資料リンクを照合済み。project moduleのimport、科学テストrunner、canonical再生成、二項確率・Q3/DP/MCは行っていない。計画に書いたgateは検証済み数値結果ではない。

## 計画段階で未確認のこと・次回への入口

保存controlとh/M/target、旧count-only計画は使えるが、**今回3候補の閉包・total追跡適合を計算又は認定していない**。提案ρ/count/frontier/transition範囲は[FINITE_SCIENCE_PLAN](FINITE_SCIENCE_PLAN.md)へ記載。kernel/DP/二項確率を起動して確認することは今回の許可外なので、候補別の未確認事項として残す。

採択済みZ1＋T1、proxy、全loss、τ、ideal IIDモデルを改めて未採択とは扱わない。今後必要なのは提案3vector・有限scopeと不足分の科学実行に対する明示許可、実行前のsource/重複再確認、callerの技術的整備である。ground/target包含/state誤差/実sampler移送/数値certificate/十分N/global RMSE/Gは引き続き未資格。

## ロードマップへの影響・STOP

統合担当への更新点は「PR-DF固定grid bridgeは完成済み、3候補の13rawと評価ルートが同定済み、既存control各1本で不足分だけ最大3 Q3の計画が公開済み」。Q3契約の採択判断待ちではなく、**実行許可・候補別gate確認待ち**である。新しいMSE/N/G成果は追加していない。D1較正はこの段階の待ち依存ではなく、適格価格/Gの後段依存として扱う。

ロードマップ本体は編集しない。計画をcommit/push・remote照合したら**mandatory STOP**。新科学実行もPF欠測補填の再実行も開始しない。
